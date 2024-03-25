import argparse
import itertools
import os
import random
import traceback
from os.path import join

import sys
sys.path.append('../')
import nibabel
import nrrd
import numpy as np
import torch
from skimage.transform import resize
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from os.path import join, exists
from config.config import data_all_path, label_all_path, index_list
from model.U_CorResNet_fix import U_CorResNet_Fix_prototype_FIM_IIM
from utils.loss_utils import Genalize_dice_loss_multi_class_3D, Proto_Contrast
from utils.utils import adjust_learning_rate_D, seed_torch, get_time, save_arg, proto_nosingle
from augmentation.volumentations import (
    Compose,
    RandomScale,
    Resize,
    RandomRotate,
    RandomFlip,
    RandomGaussianNoise,
)

def get_args():
    parser = argparse.ArgumentParser(
        description="3D Res UNet for NIH Pancreas segmentation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="input batch size for training (default: 1)",
    )
    parser.add_argument(
        "--fold", type=str, default="0,1,2,3,4", metavar="str", help="fold(0-3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2022,
        metavar="N",
        help="random seed (default: 2022)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        metavar="N",
        help="number of epochs to train (default: 300)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.9,
        metavar="PW",
        help="power for lr adjust (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=int,
        default=0.0001,
        metavar="N",
        help="weight-decay (default: 0.0001)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        metavar="N",
        help="input visible devices for training (default: 3)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        metavar="str",
        help="Optimizer (default: Adam)",
    )
    parser.add_argument(
        "--ri", type=int, default=0, metavar="int", help="random_index"
    )
    return parser.parse_args()


class PancreasDataSetAugPatch(Dataset):
    def __init__(self, fold, crop_size=(64, 120, 120), augment=True):
        self.crop_d, self.crop_h, self.crop_w = crop_size

        test_list = index_list[fold:: 5]
        train_list = []
        for index in index_list:
            if index in test_list:
                continue
            else:
                train_list.append(index)
        self.patient_index = train_list.copy()
        self.is_augment = augment
        self.aug = self.get_augmentation()
        print("{} 3d images are loaded!".format(len(self.patient_index)))

    def __len__(self):
        return len(self.patient_index)

    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((4, shape[-3], shape[-2], shape[-1]))
        background = (label == 0)
        vessel1 = (label == 1)
        vessel2 = (label == 2)
        vessel3 = (label == 3)

        results_map[0, :, :, :] = np.where(background, 1, 0)
        results_map[1, :, :, :] = np.where(vessel1, 1, 0)
        results_map[2, :, :, :] = np.where(vessel2, 1, 0)
        results_map[3, :, :, :] = np.where(vessel3, 1, 0)
        return results_map

    def pre_precessing(self, image):
        image[image <= -100] = -100
        image[image >= 240] = 240
        image += 100
        image = image / 340
        return image

    def get_augmentation(self):
        return Compose(
            [
                RandomScale((0.9, 1.1)),
                Resize(always_apply=True),
                RandomRotate((-15, 15), (-15, 15), (-15, 15)),
                RandomFlip(0),
                RandomFlip(1),
                RandomFlip(2),
                RandomGaussianNoise(),
            ],
            p=1,
        )
    def __getitem__(self, index):
        idx = self.patient_index[index]
        image = nrrd.read(
            join(data_all_path, "{}.nrrd".format(idx)))[0]

        image = image.transpose((2, 1, 0)).astype(float)

        label = nrrd.read(
            join(label_all_path, "{}.nrrd".format(idx)))[0].transpose(2, 1, 0)

        a = random.randint(0, 1)
        z = random.randint(1, 5) / 10
        if a == 0:
            path = './data/perturbed_data/V2A_%s' % str(z)
            image2 = nrrd.read(
                join(path, "{}.nrrd".format(self.patient_index[index]))
            )[0].transpose((2, 1, 0)).astype(float)
        else:
            path = './data/perturbed_data/V2D_%s' % str(z)
            image2 = nrrd.read(
                join(path, "{}.nrrd".format(self.patient_index[index]))
            )[0].transpose((2, 1, 0)).astype(float)

        # 取灰度窗和归一化
        image = self.pre_precessing(image)
        img_d, img_h, img_w = label.shape

        d_off = random.randint(0, img_d - self.crop_d)
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)

        # 取 3D Patch
        image = image[
                d_off: d_off + self.crop_d, h_off: h_off + self.crop_h, w_off: w_off + self.crop_w
                ]
        image2 = image2[
                 d_off: d_off + self.crop_d, h_off: h_off + self.crop_h, w_off: w_off + self.crop_w
                 ]
        label = label[
                d_off: d_off + self.crop_d, h_off: h_off + self.crop_h, w_off: w_off + self.crop_w
                ]

        label2 = label.copy()

        if self.is_augment:
            temp_data = {
                "image": image.transpose(1, 2, 0),
                "mask": label.transpose(1, 2, 0),
                "size": label.transpose(1, 2, 0).shape,
            }

            aug_data = self.aug(**temp_data)
            image, label = (
                aug_data["image"].transpose(2, 0, 1),
                aug_data["mask"].transpose(2, 0, 1),
            )

        if self.is_augment:
            temp_data = {
                "image": image2.transpose(1, 2, 0),
                "mask": label2.transpose(1, 2, 0),
                "size": label2.transpose(1, 2, 0).shape,
            }

            aug_data = self.aug(**temp_data)
            image2, label2 = (
                aug_data["image"].transpose(2, 0, 1),
                aug_data["mask"].transpose(2, 0, 1),
            )

        image = np.array([image]).astype(float)
        label = np.array([label]).astype(float)
        label_multi = self.id2trainId(label).astype(float)

        image2 = np.array([image2]).astype(float)
        label2 = np.array([label2]).astype(float)
        label_multi2 = self.id2trainId(label2).astype(float)

        return image.copy(), label.copy(), label_multi.copy(), image2.copy(), label2.copy(), label_multi2.copy()

def train_renji_u_conresnet_aug(args):
    # 固定随机种子
    seed_torch(args.seed)

    # 指定GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 定义网络模型
    model = U_CorResNet_Fix_prototype_FIM_IIM(in_channels=1, out_channels=4, weight_std=False).cuda()

    # 定义网络的优化器
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Loading in the Dataset
    dset_train = PancreasDataSetAugPatch(args.fold)
    train_loader = DataLoader(
        dset_train, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
    )
    print("Training Data : ", len(train_loader.dataset))

    print("#################################")
    print("fold:", args.fold)
    print("epoch:", args.epochs)
    print("lr:", args.lr)
    print("batch_size:", args.batch_size)
    print("optimizer:", args.optimizer)
    print("seed:", args.seed)
    print("#################################")

    # Defining Loss Function
    criterion_dsc = Genalize_dice_loss_multi_class_3D(channel=4)
    criterion_bce = nn.CrossEntropyLoss()
    criterion_CT  = Proto_Contrast()
    log_file_name = "F%s.txt" % (args.fold)
    if not exists(join("./checkpoints", args.time)):
        os.makedirs(join("./checkpoints", args.time))
    checkpoint_path = join("./checkpoints", args.time)
    save_arg(args, join("./checkpoints", args.time, log_file_name))

    proto_list = [[], [], [], []]
    # train for epochs
    for i in range(args.epochs):
        loss_list = []
        loss_dsc_list = []
        loss_bce_list = []
        loss_aux_list = []
        loss_CT_list = []

        with tqdm(train_loader) as t:
            for batch_idx, (image1, mask1, mask_multi1, image2, mask2, mask_multi2) in enumerate(t):
                t.set_description("Epoch%s" % i)
                image1, mask1, mask_multi1 = image1.cuda().float(), mask1.cuda().float(), mask_multi1.cuda().float()
                image2, mask2, mask_multi2 = image2.cuda().float(), mask2.cuda().float(), mask_multi2.cuda().float() # pertubed data
                try:
                    model.train()
                    optimizer.zero_grad()
                    ###########################################################
                    feature, skip1, skip2, skip3, proto_CT1, proto1, ptr1, coarse_pred1 = model.data_encoder(image1, mask1)
                    output = model.data_decoder(feature, skip1, skip2, skip3)

                    feature2, skip21, skip22, skip23, proto_CT2,_, ptr2, _ = model.data_encoder(image2, mask2, mode='aug')
                    output2 = model.data_decoder(feature2, skip21, skip22, skip23)
                    loss_dsc = criterion_dsc(torch.softmax(output, dim=1), mask_multi1) + criterion_dsc(torch.softmax(output2, dim=1), mask_multi2)
                    loss_bce = criterion_bce(output, mask1[:, 0, :, :, :].long()) + criterion_bce(output2, mask2[:, 0, :, :, :].long())
                    loss_aux = criterion_dsc(coarse_pred1, mask_multi1)
                    proto_CT = torch.cat((proto_CT1, proto_CT2), dim=0)
                    proto_ptr = torch.cat((ptr1, ptr2), dim=0)
                    proto_CT, proto_ptr = proto_nosingle(proto_CT, proto_ptr)
                    loss_CT = criterion_CT(proto_CT, proto_ptr)
                    loss_CT_list.append(loss_CT.item())

                    loss_aux_list.append(loss_aux.item())
                    loss = loss_dsc + loss_bce + loss_aux + loss_CT
                    ######################### FIM #############################
                    if i > 19:  # warm
                        # proto, coarse_pred = proto[0], coarse_pred[0]
                        feature3 = feature.clone()

                        new_proto = []
                        for j in ptr1:
                            rand = random.randint(0, 49)
                            temp_style = proto_list[j][rand]
                            new_proto.append(temp_style.unsqueeze(0))
                        new_proto = torch.cat(new_proto, dim=0)

                        B, C, D, H, W = feature3.shape
                        coarse_pred = torch.nn.Upsample(scale_factor=1/8, mode='trilinear')(coarse_pred1)
                        coarse_pred = coarse_pred.view(B, -1, D*H*W)  # B, n_classes, N

                        # 过滤掉不存在的类别
                        coarse_pred_filter = []
                        for j in range(len(ptr1)):
                            idx = ptr1[j].item()
                            coarse_pred_filter.append(coarse_pred[:, idx])
                        coarse_pred_filter = torch.stack(coarse_pred_filter, dim=1)
                        delta = torch.bmm(coarse_pred_filter.permute(0, 2, 1), (new_proto-proto1).unsqueeze(0))  # B, N, C
                        delta = delta.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

                        delta_lambda = 0.05 * (i // (0.05 * 300) + 0.5)
                        z = torch.tensor(random.uniform(0.1, 0.1 + delta_lambda))
                        z = torch.clamp(z, 0.1, 0.5)
                        feature3 = feature3 + delta * z
                        output3 = model.data_decoder(feature3, skip1, skip2, skip3)
                        loss_dsc2 = criterion_dsc(torch.softmax(output3, dim=1), mask_multi1)
                        loss_bce2 = criterion_bce(output3, mask1[:, 0, :, :, :].long())

                        loss = loss + loss_dsc2 + loss_bce2
                        loss_dsc_list.append(loss_dsc2.item())
                        loss_bce_list.append(loss_bce2.item())
                    loss_list.append(loss.item())
                    loss.backward()
                    optimizer.step()

                    # 梯度回传
                    loss_list.append(loss.item())
                    # torch.cuda.empty_cache()
                    t.set_postfix(
                        ave_loss=np.mean(loss_list),
                        aux_loss=np.mean(loss_aux_list),
                        dsc_loss=np.mean(loss_dsc_list),
                        bce_loss=np.mean(loss_bce_list),
                        CT_loss=np.mean(loss_CT_list),
                    )

                    # 将本数据的风格存入风格池
                    for j in range(len(ptr1)):
                        idx = ptr1[j].item()
                        if len(proto_list[idx]) >= 50:
                            proto_list[idx] = proto_list[idx][1:]
                        proto_list[idx].append(proto1[j].detach())

                except:
                    print(image1.size(), mask1.size(), mask_multi1.shape)
                    traceback.print_exc()
        # adjust learning rate
        adjust_learning_rate_D(optimizer, args.lr, i, args.epochs, power=args.power)

        # write train_temp and evaluate_temp records
        with open(
                join("./checkpoints", args.time, "%s" % log_file_name), "a"
        ) as f:
            f.write(
                "E %s ave_ce_loss=%s, ave_dsc_loss=%s, ave_aux_loss=%s, CT_loss=%s\n"
                % (i, np.mean(loss_bce_list), np.mean(loss_dsc_list), np.mean(loss_aux_list), np.mean(loss_CT_list))
            )

        # 保存模型
        if i > 270 and (i + 1) % 5 == 0:
            torch.save(
                model.state_dict(),
                join(
                    checkpoint_path,
                    "Network_f%s_epoch%s.pth" % (args.fold, str(i)),
                ),
            )



if __name__ == "__main__":

    args = get_args()
    args.time = "Proposed_model_lr%s_ri%s_seed%s_epoch%s" \
                % (args.lr, str(args.ri), str(args.seed), args.epochs)
    folds = list(map(int, args.fold.split(",")))

    for fold in folds:
        args.fold = fold
        train_renji_u_conresnet_aug(args)
