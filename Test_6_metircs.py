import sys
sys.path.append('../')
import argparse
import itertools
import os
from math import ceil
from os.path import join

from utils.utils import seed_torch, SR, SP

seed_torch(2022)
from skimage.morphology import skeletonize_3d, dilation
from model.U_CorResNet_fix import U_CorResNet_Fix_BL
import numpy as np
import torch

import nibabel
import nrrd
from medpy import metric


from config.config import (
    LOW_RANGE,
    HIGH_RANGE,
    data_all_path,
    label_all_path,
    index_list,
    pancreas_all_path,
    vessel_skeleton_path,
)

MARGIN = 25

def get_skeleton(label):
    skeleton_lee = np.zeros(label.shape)
    for j in range(1, 4):
        if j in label:
            temp_label = label.copy()
            temp_label[temp_label != j] = 0
            skeleton_lee += skeletonize_3d(temp_label)
    skeleton_lee = dilation(skeleton_lee, np.ones([3, 3, 3]))
    return skeleton_lee

def get_args():
    parser = argparse.ArgumentParser(
        description="Test for Res UNet on Pancreas cancer segmentation"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        metavar="N",
        help="input visible devices for training (default: 0)",
    )
    parser.add_argument(
        "--fold", type=str, default="0,1,2,3,4", metavar="str", help="fold for testing"
    )
    parser.add_argument(
        "--path",
        type=str,
        default='all',
        metavar="N",
        help="test/all, all(default)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=299,
        metavar="N",
        help="test epoch",
    )
    return parser.parse_args()

def crop_by_pancreas(image, mask, pancreas):
    arr = np.nonzero(pancreas)
    minA = max(0, min(arr[0]) - 5)
    maxA = min(len(mask), max(arr[0]) + 5)

    MARGIN = 15
    minB = max(0, min(arr[1]) - MARGIN)
    maxB = min(512, max(arr[1]) + MARGIN)
    minC = max(0, min(arr[2]) - MARGIN)
    maxC = min(512, max(arr[2]) + MARGIN)

    if (maxA - minA) % 8 != 0:
        max_A = 8 * (int((maxA - minA) / 8) + 1)
        gap = int((max_A - (maxA - minA)) / 2)
        minA = max(0, minA - gap)
        maxA = min(len(mask), minA + max_A)
        if maxA == len(mask):
            minA = maxA - max_A

    if (maxB - minB) % 8 != 0:
        max_B = 8 * (int((maxB - minB) / 8) + 1)
        gap = int((max_B - (maxB - minB)) / 2)
        minB = max(0, minB - gap)
        maxB = min(512, minB + max_B)
        if maxB == 512:
            minB = maxB - max_B

    if (maxC - minC) % 8 != 0:
        max_C = 8 * (int((maxC - minC) / 8) + 1)
        gap = int((max_C - (maxC - minC)) / 2)
        minC = max(0, minC - gap)
        maxC = min(512, minC + max_C)
        if maxC == 512:
            minC = maxC - max_C

    image, mask = (
        image[minA:maxA, minB:maxB, minC:maxC],
        mask[minA:maxA, minB:maxB, minC:maxC],
    )

    bbox = [minA, maxA, minB, maxB, minC, maxC]

    return image, mask, bbox
def pad_image(img, target_size):
    """Pad an image up to the target size."""
    deps_missing = target_size[0] - img.shape[2]
    rows_missing = target_size[1] - img.shape[3]
    cols_missing = target_size[2] - img.shape[4]
    padded_img = np.pad(
        img,
        ((0, 0), (0, 0), (0, deps_missing), (0, rows_missing), (0, cols_missing)),
        "constant",
    )
    return padded_img

def pred_result(test, model):
    with torch.no_grad():
        padded_img = torch.from_numpy(test).cuda().float()
        result = model(padded_img)[0]
        padded_prediction = torch.softmax(result, dim=1).cpu().detach().numpy()
        padded_prediction = np.squeeze(padded_prediction)

    return padded_prediction

def predict_sliding(net, images, tile_size, classes, overlap=1 / 3):

    image = np.expand_dims(images, axis=0)
    assert len(image.shape) == 5

    # B*C*D*H*W
    image_size = image.shape

    # 2/3 * (64,120,120)
    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))

    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)

    full_probs = np.zeros(
        (classes, image_size[2], image_size[3], image_size[4])
    ).astype(float)
    count_predictions = np.zeros(
        (classes, image_size[2], image_size[3], image_size[4])
    ).astype(float)

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                y1 = int(row * strideHW)
                x1 = int(col * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                y2 = min(y1 + tile_size[1], image_size[3])
                x2 = min(x1 + tile_size[2], image_size[4])
                d1 = max(int(d2 - tile_size[0]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                padded_img = pad_image(img, tile_size)

                # B*C*D*H*W

                padded_prediction = pred_result(padded_img, net)

                prediction = padded_prediction[:, 0: img.shape[-3], 0: img.shape[-2], 0: img.shape[-1]]

                count_predictions[:, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs

def evaluate(model, test_list, args, fold, patch_size=None):
    DSC_list = [[], [], []]
    hd_list = [[], [], []]
    assd_list = [[], [], []]
    mcd_list = [[], [], []]
    SR_list = [[], [], []]
    SP_list = [[], [], []]
    # d, h, w = 64, 120, 120
    input_size = patch_size

    for p in test_list:

        image = nibabel.load(
            join(args.datapath, "{}.nii.gz".format(p))
        )
        pixdim = image.header['pixdim']
        voxelspacing = [pixdim[3], pixdim[2], pixdim[1]]
        image = image.get_fdata()
        if len(image.shape) == 5:
            image = image[:, :, :, 0, 0]

        image = image.transpose((2, 1, 0)).astype(float)

        mask = nrrd.read(
            join(args.labelpath, "{}.nrrd".format(p))
        )[0].transpose((2, 1, 0))

        mask_pancreas = nrrd.read(
            join(args.pancreaspath, "{}.nrrd".format(p))
        )[0].transpose((2, 1, 0))

        pred_3D = np.zeros_like(mask)

        image, mask_crop, bbox = crop_by_pancreas(image, mask, mask_pancreas)
        mask = mask[bbox[0]: bbox[1], bbox[2]: bbox[3], bbox[4]:bbox[5]]

        mask_skeleton = nrrd.read(
            join(args.mask_skeleton, "{}.nrrd".format(p))
        )[0].transpose(2, 1, 0).astype(np.uint8)
        mask_skeleton = mask_skeleton[bbox[0]: bbox[1], bbox[2]: bbox[3], bbox[4]:bbox[5]]

        if np.max(image > 1):
            np.minimum(np.maximum(image, LOW_RANGE, image), HIGH_RANGE, image)
            image -= LOW_RANGE
            image /= HIGH_RANGE - LOW_RANGE

        image = np.array([image])

        # B*D*H*W
        image = image.astype(float)

        pred_patient = predict_sliding(
            model, image, input_size, classes=4, overlap=float(1 / 3)
        )

        pred_patient = np.argmax(pred_patient, axis=0)
        pred_patient = np.squeeze(pred_patient)

        pred_3D[bbox[0]: bbox[1], bbox[2]: bbox[3], bbox[4]: bbox[5]] = pred_patient
        pred_skeleton = get_skeleton(pred_patient)
        with open(
                join(
                    args.path,
                    "patient_allmetircs_plus.txt",
                ),
                "a",
        ) as f:
            f.write("patient%s " % (p))
        for j in range(1, 4):
            temp_pred = pred_patient.copy()
            temp_pred[temp_pred != j] = 0
            temp_mask = mask.copy()
            temp_mask[temp_mask != j] = 0
            temp_mask_skeleton = mask_skeleton.copy()
            temp_mask_skeleton[temp_mask_skeleton != j] = 0
            temp_pred_skeleton = pred_skeleton.copy()
            temp_pred_skeleton[temp_pred_skeleton != j] = 0
            if np.sum(temp_mask) == 0:
                continue
            temp_patient_dsc_full = metric.dc(temp_pred, temp_mask)
            DSC_list[j - 1].append(temp_patient_dsc_full)
            temp_sr = SR(temp_pred, temp_mask_skeleton)
            SR_list[j - 1].append(temp_sr)
            temp_sp = SP(temp_pred_skeleton, temp_mask)
            SP_list[j - 1].append(temp_sp)
            if np.sum(temp_pred) == 0:
                temp_patient_hd_full = 0
                temp_patient_assd_full = 0
                temp_patient_mcd_full = 0
                with open(
                        join(
                            args.path,
                            "patient_allmetircs_plus.txt",
                        ),
                        "a",
                ) as f:
                    f.write(
                        "Pred no vessel %s\n" % (str(j)))
                print("Pred no vessel %s" % (str(j)))
            else:
                temp_patient_hd_full = metric.hd(temp_pred, temp_mask, voxelspacing=voxelspacing)
                temp_patient_assd_full = metric.assd(temp_pred, temp_mask, voxelspacing=voxelspacing)
                hd_list[j - 1].append(temp_patient_hd_full)
                assd_list[j - 1].append(temp_patient_assd_full)
                if np.sum(temp_pred_skeleton) == 0 or np.sum(temp_mask_skeleton) == 0:
                    temp_patient_mcd_full = 'no vessel'
                else:
                    temp_patient_mcd_full = metric.assd(temp_pred_skeleton, temp_mask_skeleton, voxelspacing=voxelspacing)
                    mcd_list[j - 1].append(temp_patient_mcd_full)

            print("vessel %s: dsc %.8s, hd %.8s, assd %.8s, mcd %.8s, sr %.8s, sp %.8s,"
                  % (str(j), temp_patient_dsc_full, temp_patient_hd_full, temp_patient_assd_full, temp_patient_mcd_full,
                     temp_sr, temp_sp))

            with open(
                    join(
                        args.path,
                        "patient_allmetircs_plus.txt",
                    ),
                    "a",
            ) as f:
                f.write(
                    "vessel %s: dsc %.8s, hd %.8s, assd %.8s, mcd %.8s, sr %.8s, sp %.8s\n" % (
                        str(j), temp_patient_dsc_full, temp_patient_hd_full, temp_patient_assd_full,
                        temp_patient_mcd_full, temp_sr, temp_sp))
        print('*' * 30)
        with open(
                join(
                    args.path,
                    "patient_allmetircs_plus.txt",
                ),
                "a",
        ) as f:
            f.write('*' * 30 + '\n')

    for k in range(1, 4):
        print('Fold%s: vessel %.8s Average: dsc %.8s, hd %.8s, assd %.8s, mcd %.8s, sr %.8s, sp %.8s'
              % (str(fold), str(k), str(np.mean(DSC_list[k - 1])), str(np.mean(hd_list[k - 1])),
                 str(np.mean(assd_list[k - 1])), str(np.mean(mcd_list[k - 1])), str(np.mean(SR_list[k-1])),
                 str(np.mean(SP_list[k-1]))))

    with open(
            join(
                args.path,
                "patient_allmetircs_plus.txt",
            ),
            "a",
    ) as f:
        f.write(
            'Fold: %s, Vessel 1: DSC %.8s, HD %.8s, ASSD %.8s, MCD %.8s, sr %.8s, sp %.8s\n'
            % (str(fold), str(np.mean(DSC_list[0])), str(np.mean(hd_list[0])), str(np.mean(assd_list[0])),
               str(np.mean(mcd_list[0])), np.mean(SR_list[0]),  np.mean(SP_list[0])))
        f.write(
            'Fold: %s, Vessel 2: DSC %.8s, HD %.8s, ASSD %.8s, MCD %.8s, sr %.8s, sp %.8s\n'
            % (str(fold), str(np.mean(DSC_list[1])), str(np.mean(hd_list[1])), str(np.mean(assd_list[1])),
               str(np.mean(mcd_list[1])), np.mean(SR_list[1]),  np.mean(SP_list[1])))
        f.write(
            'Fold: %s, Vessel 3: DSC %.8s, HD %.8s, ASSD %.8s, MCD %.8s, sr %.8s, sp %.8s\n'
            % (str(fold), str(np.mean(DSC_list[2])), str(np.mean(hd_list[2])), str(np.mean(assd_list[2])),
               str(np.mean(mcd_list[2])), np.mean(SR_list[2]),  np.mean(SP_list[2])))
    vessel1 = np.asarray([[np.mean(DSC_list[0]), np.mean(hd_list[0]), np.mean(assd_list[0]), np.mean(mcd_list[0]),
                           np.mean(SR_list[0]), np.mean(SP_list[0])],
                          [np.std(DSC_list[0]), np.std(hd_list[0]), np.std(assd_list[0]), np.std(mcd_list[0]),
                           np.std(SR_list[0]), np.std(SP_list[0])]])
    vessel2 = np.asarray([[np.mean(DSC_list[1]), np.mean(hd_list[1]), np.mean(assd_list[1]), np.mean(mcd_list[1]),
                           np.mean(SR_list[1]), np.mean(SP_list[1])],
                          [np.std(DSC_list[1]), np.std(hd_list[1]), np.std(assd_list[1]), np.std(mcd_list[1]),
                           np.std(SR_list[1]), np.std(SP_list[1])]])
    vessel3 = np.asarray([[np.mean(DSC_list[2]), np.mean(hd_list[2]), np.mean(assd_list[2]), np.mean(mcd_list[2]),
                           np.mean(SR_list[2]), np.mean(SP_list[2])],
                          [np.std(DSC_list[2]), np.std(hd_list[2]), np.std(assd_list[2]), np.std(mcd_list[2]),
                           np.std(SR_list[2]), np.std(SP_list[2])]])
    return vessel1, vessel2, vessel3

if __name__ == "__main__":
    args = get_args()
    args.seed = 2022
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.fold = -1
    args.logs = True
    args.save = True
    args.ri = 1

    args.datapath = data_all_path
    args.labelpath = label_all_path
    args.pancreaspath = pancreas_all_path
    args.mask_skeleton = vessel_skeleton_path

    args.model_name = "Network_epoch%s.pth" % str(args.epoch)
    model = U_CorResNet_Fix_BL(in_channels=1, out_channels=4).cuda().eval()
    model_state = model.state_dict()
    data = []

    vessel1 = np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    vessel2 = np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    vessel3 = np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    count = 0
    for fold in range(5):
        if args.fold != -1:
            if fold != args.fold:
                continue
            test_list = index_list[fold:: 5]
        print('fold%s' % str(fold))
        print(len(test_list), test_list)
        model_path = join(args.path, args.model_name.replace("fold", str(fold)))
        train_state_dict = torch.load(model_path)
        # 只留下预测要用的权重
        state_dict = {k: v for k, v in train_state_dict.items() if k in model_state.keys()}
        model.load_state_dict(state_dict)
        temp_vessel1, temp_vessel2, temp_vessel3 = evaluate(model, test_list, args, fold, patch_size=(64, 120, 120))
        vessel1 = vessel1 + temp_vessel1
        vessel2 = vessel2 + temp_vessel2
        vessel3 = vessel3 + temp_vessel3
        count += 1

    vessel1 = np.asarray(vessel1) / count
    vessel2 = np.asarray(vessel2) / count
    vessel3 = np.asarray(vessel3) / count
    if count == 5:
        with open(
                join(
                    args.path,
                    "patient_allmetircs_plus.txt",
                ),
                "a",
        ) as f:
            f.write('5-Fold_Average: DSC HD ASSD MCD SR SP\n')
            f.write('vessel1 mean: %s %s %s %s %s %s\n' % (str(vessel1[0][0]), str(vessel1[0][1]), str(vessel1[0][2]),
                                                           str(vessel1[0][3]), str(vessel1[0][4]), str(vessel1[0][5])))
            f.write('vessel2 mean: %s %s %s %s %s %s\n' % (str(vessel2[0][0]), str(vessel2[0][1]), str(vessel2[0][2]),
                                                           str(vessel2[0][3]), str(vessel2[0][4]), str(vessel2[0][5])))
            f.write('vessel3 mean: %s %s %s %s %s %s\n' % (str(vessel3[0][0]), str(vessel3[0][1]), str(vessel3[0][2]),
                                                           str(vessel3[0][3]), str(vessel3[0][4]), str(vessel3[0][5])))
            f.write('vessel1 std: %s %s %s %s %s %s\n' % (str(vessel1[1][0]), str(vessel1[1][1]), str(vessel1[1][2]),
                                                           str(vessel1[1][3]), str(vessel1[1][4]), str(vessel1[1][5])))
            f.write('vessel2 std: %s %s %s %s %s %s\n' % (str(vessel2[1][0]), str(vessel2[1][1]), str(vessel2[1][2]),
                                                          str(vessel2[1][3]), str(vessel2[1][4]), str(vessel2[1][5])))
            f.write('vessel3 std: %s %s %s %s %s %s\n' % (str(vessel3[1][0]), str(vessel3[1][1]), str(vessel3[1][2]),
                                                          str(vessel3[1][3]), str(vessel3[1][4]), str(vessel3[1][5])))

