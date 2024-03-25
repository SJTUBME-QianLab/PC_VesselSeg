#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Time    : 2020/8/12 16:08
# @Author  : Dirk Li
# @Email   : junli.dirk@gmail.com
# @File    : criterion.py
# @Software: PyCharm,Windows10
# @Hardware: 32G-RAM,Intel-i7-7700k,GTX-1080

import os
import random
import time
import math
import numpy as np
import torch
from skimage import measure
import yaml
import torch.nn as nn
from torch.nn import functional as F

def DSC_computation(label, pred):
    pred_sum = pred.sum()
    label_sum = label.sum()
    inter_sum = np.logical_and(pred, label).sum()
    return (2 * float(inter_sum) + 1e-5) / (pred_sum + label_sum + 1e-5)


def seed_torch(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate_D(optimizer, learning_rate, i_iter, max_iter, power=0.9):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    optimizer.param_groups[0]["lr"] = lr


def adjust_learning_rate(opts, base_lr, iter, max_iter, power=0.9):
    lr = lr_poly(base_lr, iter, max_iter, power)
    for opt in opts:
        opt.param_groups[0]["lr"] = lr
        if len(opt.param_groups) > 1:
            opt.param_groups[1]["lr"] = lr * 10


def get_time():
    cur_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time()))
    return cur_time

def center_crop(data, height, width):
    assert (
        data.shape[1] >= height
    ), "Error! The height %s should be smaller than data %s!" % (height, data.shape)
    assert (
        data.shape[2] >= width
    ), "Error! The width %s should be smaller than data %s!" % (height, data.shape)

    height_data, width_data = data.shape[1], data.shape[2]

    s_h = int((height_data - height) / 2)
    s_w = int((width_data - width) / 2)

    return data[:, s_h: s_h + height, s_w: s_w + width]

def save_arg(args, path):
    arg_dict = vars(args)
    with open(path, "a") as f:
        yaml.dump(arg_dict, f)


def padding_to_fit_unet(bbox, H, W):
    if (bbox[1] - bbox[0]) % 16 != 0:
        target_length = ((bbox[1] - bbox[0]) // 16 + 1) * 16
        temp_gap = target_length - (bbox[1] - bbox[0])
        left = int(temp_gap / 2)
        right = temp_gap - left
        bbox[0] = int(max(bbox[0] - left, 0))
        bbox[1] = int(min(bbox[1] + right, H))
        if bbox[0] == 0:
            bbox[1] = bbox[0] + target_length
        elif bbox[1] == H:
            bbox[0] = bbox[1] - target_length

    if (bbox[3] - bbox[2]) % 16 != 0:
        target_length = ((bbox[3] - bbox[2]) // 16 + 1) * 16
        temp_gap = target_length - (bbox[3] - bbox[2])
        left = int(temp_gap / 2)
        right = temp_gap - left
        bbox[2] = int(max(bbox[2] - left, 0))
        bbox[3] = int(min(bbox[3] + right, W))
        if bbox[2] == 0:
            bbox[3] = bbox[2] + target_length
        elif bbox[3] == W:
            bbox[2] = bbox[3] - target_length

    return bbox

def padding_z(data, label=None, mode="down"):
    if mode == "down":
        if label is not None:
            slices = len(data)
            new_slice = int(slices / 32) * 32
            new_data = np.zeros((new_slice, data.shape[1], data.shape[2]))
            new_label = np.zeros((new_slice, data.shape[1], data.shape[2]))
            if new_slice >= slices:
                up = int((new_slice - slices) / 2)
                new_data[up: up + slices] = data
                new_label[up: up + slices] = label
                return new_data, new_label
            else:
                up = int((slices - new_slice) / 2)
                new_data = data[up: up + new_slice]
                new_label = label[up: up + new_slice]
                return new_data, new_label
        else:
            slices = len(data)
            new_slice = int(slices / 32) * 32
            new_data = np.zeros((new_slice, data.shape[1], data.shape[2]))
            if new_slice >= slices:
                up = int((new_slice - slices) / 2)
                new_data[up : up + slices] = data
                return new_data
            else:
                up = int((slices - new_slice) / 2)
                new_data = data[up : up + new_slice]
                return new_data

    elif mode == "up":
        if label is not None:
            slices = len(data)
            new_slice = int(slices / 32) * 32 + 32
            new_data = np.zeros((new_slice, data.shape[1], data.shape[2]))
            new_label = np.zeros((new_slice, data.shape[1], data.shape[2]))
            if new_slice >= slices:
                up = int((new_slice - slices) / 2)
                new_data[up : up + slices] = data
                new_label[up : up + slices] = label
                return new_data, new_label
            else:
                up = int((slices - new_slice) / 2)
                new_data = data[up : up + new_slice]
                new_label = label[up : up + new_slice]
                return new_data, new_label
        else:
            slices = len(data)
            new_slice = int(slices / 32) * 32 + 32
            new_data = np.zeros((new_slice, data.shape[1], data.shape[2]))
            if new_slice >= slices:
                up = int((new_slice - slices) / 2)
                new_data[up : up + slices] = data
                return new_data
            else:
                up = int((slices - new_slice) / 2)
                new_data = data[up : up + new_slice]
                return new_data

def unpadding_z(pred, label):
    slice_label = len(label)
    slice_pred = len(pred)

    new_pred = np.zeros_like(label).astype(np.float)

    if slice_pred >= slice_label:
        up = int((slice_pred - slice_label) / 2)
        new_pred[:] = pred[up : up + slice_label]
        return new_pred
    else:
        up = int((slice_label - slice_pred) / 2)
        new_pred[up: up + slice_pred] = pred
        return new_pred

def crop(data, label, margin=20):
    arr = np.nonzero(label)
    H, W = label.shape[1], label.shape[2]
    minA = min(arr[1])
    maxA = max(arr[1])
    minB = min(arr[2])
    maxB = max(arr[2])
    bbox = [int(max(minA - margin, 0)), int(min(maxA + margin + 1, H)), int(max(minB - margin, 0)),
            int(min(maxB + margin + 1, W))]

    bbox = padding_to_fit_unet(bbox, H, W)

    cropped_image = data[:, bbox[0]: bbox[1], bbox[2]: bbox[3]]
    cropped_label = label[:, bbox[0]: bbox[1], bbox[2]: bbox[3]]

    return cropped_image, cropped_label, bbox


def center_uncrop(pred, label):
    assert (
        label.shape[1] >= pred.shape[1]
    ), "Error! The pred-height should be smaller than label!"
    assert (
        label.shape[2] >= pred.shape[2]
    ), "Error! The pred-width should be smaller than label!"

    height, width = pred.shape[1], pred.shape[2]
    height_label, width_label = label.shape[1], label.shape[2]

    s_h = int((height_label - height) / 2)
    s_w = int((width_label - width) / 2)

    new_pred = np.zeros_like(label).astype(np.float)
    new_pred[:, s_h : s_h + height, s_w : s_w + width] = pred

    return new_pred

def uncrop(ori_data, target_data, bbox):
    temp = np.zeros_like(target_data)
    temp[:, bbox[0]: bbox[1], bbox[2]: bbox[3]] = ori_data
    return temp


def post_processing_soft(F, S, threshold=0.5, top2=1):
    F_sum = F.sum()
    if F_sum == 0:
        return F
    if F_sum >= np.product(F.shape) / 2:
        return F
    height = F.shape[0]
    width = F.shape[1]
    depth = F.shape[2]
    ll = np.array(np.nonzero(S))
    marked = np.zeros(F.shape, dtype=np.bool)
    queue = np.zeros((F_sum, 3), dtype=np.int)
    volume = np.zeros(F_sum, dtype=np.int)
    head = 0
    tail = 0
    bestHead = 0
    bestTail = 0
    bestHead2 = 0
    bestTail2 = 0
    for l in range(ll.shape[1]):
        if not marked[ll[0, l], ll[1, l], ll[2, l]]:
            temp = head
            marked[ll[0, l], ll[1, l], ll[2, l]] = True
            queue[tail, :] = [ll[0, l], ll[1, l], ll[2, l]]
            tail = tail + 1
            while (head < tail):
                t1 = queue[head, 0]
                t2 = queue[head, 1]
                t3 = queue[head, 2]
                if t1 > 0 and F[t1 - 1, t2, t3] and not marked[t1 - 1, t2, t3]:
                    marked[t1 - 1, t2, t3] = True
                    queue[tail, :] = [t1 - 1, t2, t3]
                    tail = tail + 1
                if t1 < height - 1 and F[t1 + 1, t2, t3] and not marked[t1 + 1, t2, t3]:
                    marked[t1 + 1, t2, t3] = True
                    queue[tail, :] = [t1 + 1, t2, t3]
                    tail = tail + 1
                if t2 > 0 and F[t1, t2 - 1, t3] and not marked[t1, t2 - 1, t3]:
                    marked[t1, t2 - 1, t3] = True
                    queue[tail, :] = [t1, t2 - 1, t3]
                    tail = tail + 1
                if t2 < width - 1 and F[t1, t2 + 1, t3] and not marked[t1, t2 + 1, t3]:
                    marked[t1, t2 + 1, t3] = True
                    queue[tail, :] = [t1, t2 + 1, t3]
                    tail = tail + 1
                if t3 > 0 and F[t1, t2, t3 - 1] and not marked[t1, t2, t3 - 1]:
                    marked[t1, t2, t3 - 1] = True
                    queue[tail, :] = [t1, t2, t3 - 1]
                    tail = tail + 1
                if t3 < depth - 1 and F[t1, t2, t3 + 1] and not marked[t1, t2, t3 + 1]:
                    marked[t1, t2, t3 + 1] = True
                    queue[tail, :] = [t1, t2, t3 + 1]
                    tail = tail + 1
                head = head + 1
            if tail - temp > bestTail - bestHead:
                bestHead2 = bestHead
                bestTail2 = bestTail
                bestHead = temp
                bestTail = tail
            elif tail - temp > bestTail2 - bestHead2:
                bestHead2 = temp
                bestTail2 = tail
            volume[temp: tail] = tail - temp
    volume = volume[0: tail]
    if top2:
        target_voxel = np.where(volume >= (bestTail2 - bestHead2) * threshold)
    else:
        target_voxel = np.where(volume >= (bestTail - bestHead) * threshold)
    F0 = np.zeros(F.shape, dtype=np.bool)
    F0[tuple(map(tuple, np.transpose(queue[target_voxel, :])))] = True
    return F0


def post_processing(labels, r=0.5):
    labels = measure.label(labels, connectivity=3)

    # 找到最大体积块和相应的标签，分别是max_num和max_pixel
    max_num = 0
    max_pixel = 1
    for j in range(1, np.max(labels) + 1):
        if np.sum(labels == j) > max_num:
            max_num = np.sum(labels == j)
            max_pixel = j

    # 若仅保留最大体积块，可以注释以下三行
    for j in range(1, np.max(labels) + 1):
        if np.sum(labels == j) > r * np.sum(labels == max_pixel):
            labels[labels == j] = max_pixel

    labels[labels != max_pixel] = 0
    labels[labels == max_pixel] = 1

    return labels

def get_one_hot(label, N):
    size = list(label.size())
    label = label.contiguous().view(-1).long()  # reshape 为向量
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, label)  # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)

def get_one_hot_cpu(label, N):
    size = list(label.size())
    label = label.contiguous().view(-1).long()  # reshape 为向量
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)  # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)


def crop_data_label_from_50(data, label):
    count = []
    for i in range(label.shape[0]):
        if label[i].sum() > 50:
            count.append(i)
    return data[count], label[count], count

def unpaading_z(pred, label):
    slices = len(label)
    new_slice = len(pred)

    if slices == new_slice:
        return pred

    else:
        new_label = np.zeros_like(label).astype(np.float32)
        up = int((new_slice - slices) / 2)
        new_label[:] = pred[up:up + slices]
        return new_label

class dice_loss_small_class_3D(nn.Module):
    def __init__(self, channel=5):
        super(dice_loss_small_class_3D, self).__init__()
        self.loss_lambda = [1, 1, 1, 4, 4]
        self.channel = channel

    def forward(self, logits, gt):
        dice = 0
        eps = 1e-7

        assert len(logits.shape) == 5, 'This loss is for 3D data (BCDHW), please check your output!'

        softmaxpred = logits

        for i in range(self.channel):
            inse = torch.sum(softmaxpred[:, i, :, :, :] * gt[:, i, :, :, :])
            l = torch.sum(softmaxpred[:, i, :, :, :])
            r = torch.sum(gt[:, i, :, :, :])
            dice += ((inse + eps) / (l + r + eps)) * self.loss_lambda[i] / sum(self.loss_lambda)

        return 1 - 2.0 * dice / self.channel

class dice_loss_big_class_3D(nn.Module):
    def __init__(self, channel=5):
        super(dice_loss_big_class_3D, self).__init__()
        self.loss_lambda = [1, 4, 4, 1, 1]
        self.channel = channel

    def forward(self, logits, gt):
        dice = 0
        eps = 1e-7

        assert len(logits.shape) == 5, 'This loss is for 3D data (BCDHW), please check your output!'

        softmaxpred = logits

        for i in range(self.channel):
            inse = torch.sum(softmaxpred[:, i, :, :, :] * gt[:, i, :, :, :])
            l = torch.sum(softmaxpred[:, i, :, :, :])
            r = torch.sum(gt[:, i, :, :, :])
            dice += ((inse + eps) / (l + r + eps)) * self.loss_lambda[i] / sum(self.loss_lambda)

        return 1 - 2.0 * dice / self.channel

def cal_recall(pred, label):
    return np.logical_and(pred, label).sum() / label.sum()


def cal_precision(pred, label):
    return np.logical_and(pred, label).sum() / pred.sum()

def SR(result, reference_skeleton):
    # skeleton recall
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference_skeleton.astype(np.bool))

    intersection = np.count_nonzero(result & reference)

    size_i2 = np.count_nonzero(reference)

    try:
        SR = intersection / float(size_i2)
    except ZeroDivisionError:
        SR = 0.0

    return SR

def SP(result_skeleton, reference):
    # skeleton precision
    result = np.atleast_1d(result_skeleton.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)

    try:
        SP = intersection / float(size_i1)
    except ZeroDivisionError:
        SP = 0.0
    return SP

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2), dim=0)

def proto_equalize(short_ptr, long_ptr, long_proto):
    dele_ptr = [long_ptr.tolist().index(i) for i in long_ptr if i not in short_ptr]
    dele_ptr = sorted(dele_ptr, reverse=True)
    for i in dele_ptr:
        long_ptr = del_tensor_ele(long_ptr, i)
        long_proto = del_tensor_ele(long_proto, i)
    return long_ptr, long_proto

def proto_simap_equalize(short_ptr, long_ptr, long_proto, sim_map):
    dele_ptr = [long_ptr.tolist().index(i) for i in long_ptr if i not in short_ptr]
    dele_ptr = sorted(dele_ptr, reverse=True)
    for i in dele_ptr:
        long_ptr = del_tensor_ele(long_ptr, i)
        long_proto = del_tensor_ele(long_proto, i)
        sim_map = del_tensor_ele(sim_map, i)
    return long_ptr, long_proto, sim_map

def proto_nosingle(proto, ptr):
    index_list = torch.unique(ptr)
    ptr_list = ptr.tolist()
    dele_ptr = [ptr_list.index(i) for i in index_list if ptr_list.count(i) == 1]
    dele_ptr = sorted(dele_ptr, reverse=True)
    for i in dele_ptr:
        ptr = del_tensor_ele(ptr, i)
        proto = del_tensor_ele(proto, i)
    return proto, ptr

def proto_sim_nosigle(proto, simmap, ptr):
    index_list = torch.unique(ptr)
    ptr_list = ptr.tolist()
    dele_ptr = [ptr_list.index(i) for i in index_list if ptr_list.count(i) == 1]
    dele_ptr = sorted(dele_ptr, reverse=True)
    for i in dele_ptr:
        ptr = del_tensor_ele(ptr, i)
        proto = del_tensor_ele(proto, i)
        simmap = del_tensor_ele(simmap, i)
    return proto, ptr, simmap

def get_prototype_hardeasy(features, labels, preds):
    '''
    :return: easy_prototype, hard prototype
    '''
    features = F.interpolate(features, size=labels.shape[-3:], mode='trilinear')
    features = features.contiguous().view(-1, features.shape[1])  # [N,C]
    labels = labels.contiguous().view(-1)  # [N]
    preds = preds.contiguous().view(-1)  # [N]

    label_idx = torch.unique(labels)
    label_idx = label_idx.long()
    prototype = []
    proto_ptr = []
    for i in label_idx:
        hard_indices = ((labels == i) & (preds != i)).nonzero()
        easy_indices = ((labels == i) & (preds == i)).nonzero()
        if len(hard_indices != 0):
            hard_features = features[hard_indices, :].squeeze(1)
            hard_proto = torch.mean(hard_features, dim=0, keepdim=True)
            prototype.append(hard_proto)
            proto_ptr.append(i.unsqueeze(0))

        if len(easy_indices != 0):
            easy_features = features[easy_indices, :].squeeze(1)
            easy_proto = torch.mean(easy_features, dim=0, keepdim=True)
            prototype.append(easy_proto)
            proto_ptr.append(i.unsqueeze(0))

    prototype = torch.cat(prototype, dim=0)  # n*c
    proto_ptr = torch.cat(proto_ptr, dim=0)  # n*c

    return prototype, proto_ptr

def change_model_state_dict(model_dict, pretrain_model):
    encoder_dict = {'encoder.' + k: v for k, v in pretrain_model.items() if ('encoder.') + k in model_dict.keys()}
    decoder_dict = {'decoder.' + k: v for k, v in pretrain_model.items() if ('decoder.') + k in model_dict.keys()}
    state_dict = {}
    state_dict.update(encoder_dict)
    state_dict.update(decoder_dict)

    return state_dict

if __name__ == "__main__":

    output = torch.randn(2, 3, 5, 5)
    print(output)
    output = nn.functional.softmax(output, dim=1).numpy()
    output = np.argmax(output, 1)
    print(output)
    output[output != 1] = 0
    print(output)
