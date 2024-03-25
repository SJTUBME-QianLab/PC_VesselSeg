import torch
import torch.nn as nn
from torch.nn import functional as F

from model.Upsample import UpsampleDeterministicP3D

affine_par = True
inplace = False

import os
import random

class Conv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        groups=1,
        bias=False,
    ):
        super(Conv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = (
            weight.mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
            .mean(dim=4, keepdim=True)
        )
        weight = weight - weight_mean
        std = torch.sqrt(
            torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12
        ).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def conv3x3x3(
    in_planes,
    out_planes,
    kernel_size=(3, 3, 3),
    stride=(1, 1, 1),
    padding=(1, 1, 1),
    dilation=(1, 1, 1),
    bias=False,
    weight_std=False,
):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
    else:
        return nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )


class NoBottleneck(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        downsample=None,
        fist_dilation=1,
        multi_grid=1,
        weight_std=False,
    ):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.relu = nn.ReLU(inplace=True)

        self.gn1 = nn.GroupNorm(8, inplanes)
        self.conv1 = conv3x3x3(
            inplanes,
            planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=dilation * multi_grid,
            dilation=dilation * multi_grid,
            bias=False,
            weight_std=self.weight_std,
        )

        self.gn2 = nn.GroupNorm(8, planes)
        self.conv2 = conv3x3x3(
            planes,
            planes,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=dilation * multi_grid,
            dilation=dilation * multi_grid,
            bias=False,
            weight_std=self.weight_std,
        )

        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        skip = x

        seg = self.gn1(x)
        seg = self.relu(seg)
        seg = self.conv1(seg)

        seg = self.gn2(seg)
        seg = self.relu(seg)
        seg = self.conv2(seg)

        if self.downsample is not None:
            skip = self.downsample(x)

        seg = seg + skip
        return seg

def _make_layer(
    block,
    inplanes,
    outplanes,
    blocks,
    stride=(1, 1, 1),
    dilation=(1, 1, 1),
    multi_grid=1,
    weight_std=False,
):
    downsample = None
    if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != outplanes:
        downsample = nn.Sequential(
            nn.GroupNorm(8, inplanes),
            nn.ReLU(inplace=True),
            conv3x3x3(
                inplanes,
                outplanes,
                kernel_size=(1, 1, 1),
                stride=stride,
                padding=(0, 0, 0),
                weight_std=weight_std,
            ),
        )

    layers = []
    generate_multi_grid = (
        lambda index, grids: grids[index % len(grids)]
        if isinstance(grids, tuple)
        else 1
    )
    layers.append(
        block(
            inplanes,
            outplanes,
            stride,
            dilation=dilation,
            downsample=downsample,
            multi_grid=generate_multi_grid(0, multi_grid),
            weight_std=weight_std,
        )
    )

    for i in range(1, blocks):
        layers.append(
            block(
                inplanes,
                outplanes,
                dilation=dilation,
                multi_grid=generate_multi_grid(i, multi_grid),
                weight_std=weight_std,
            )
        )
    return nn.Sequential(*layers)

class Unet_Encoder(nn.Module):
    def __init__(self, in_channels, weight_std=False):
        super(Unet_Encoder, self).__init__()

        layers = [1, 2, 2, 2, 2]
        self.weight_std = weight_std

        self.conv_4_32 = nn.Sequential(
            conv3x3x3(
                in_channels,
                32,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                weight_std=self.weight_std,
            )
        )

        self.conv_32_64 = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            conv3x3x3(
                32,
                64,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                weight_std=self.weight_std,
            ),
        )

        self.conv_64_128 = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            conv3x3x3(
                64,
                128,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                weight_std=self.weight_std,
            ),
        )

        self.conv_128_256 = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            conv3x3x3(
                128,
                256,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                weight_std=self.weight_std,
            ),
        )

        self.layer0 = _make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1, 1), weight_std=self.weight_std)
        self.layer1 = _make_layer(NoBottleneck, 64, 64, layers[1], stride=(1, 1, 1), weight_std=self.weight_std)
        self.layer2 = _make_layer(NoBottleneck, 128, 128, layers[2], stride=(1, 1, 1), weight_std=self.weight_std)
        self.layer3 = _make_layer(NoBottleneck, 256, 256, layers[3], stride=(1, 1, 1), weight_std=self.weight_std)
        self.layer4 = _make_layer(NoBottleneck, 256, 256, layers[4], stride=(1, 1, 1), dilation=(2, 2, 2)
                                  , weight_std=self.weight_std)

    def forward(self, x):
        self.shape = [x.shape[-3], x.shape[-2], x.shape[-1]]
        x = self.conv_4_32(x)
        x = self.layer0(x)
        skip1 = x

        x = self.conv_32_64(x)
        x = self.layer1(x)
        skip2 = x

        x = self.conv_64_128(x)
        x = self.layer2(x)
        skip3 = x

        x = self.conv_128_256(x)
        x = self.layer3(x)

        x = self.layer4(x)

        return x, skip1, skip2, skip3

class Unet_Decoder(nn.Module):
    def __init__(self, out_channels, weight_std=False):
        super(Unet_Decoder, self).__init__()
        self.fusionConv = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            conv3x3x3(
                256,
                128,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                dilation=(1, 1, 1),
                weight_std=weight_std,
            ),
        )

        self.seg_x4 = _make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1, 1), weight_std=weight_std)
        self.seg_x2 = _make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1, 1), weight_std=weight_std)
        self.seg_x1 = _make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1, 1), weight_std=weight_std)

        self.seg_cls = nn.Sequential(nn.Conv3d(32, out_channels, kernel_size=1))

    def forward(self, x, skip1, skip2, skip3):
        x = self.fusionConv(x)

        ## seg-decoder
        seg_x4 = UpsampleDeterministicP3D(2)(x)
        seg_x4 = seg_x4 + skip3
        seg_x4 = self.seg_x4(seg_x4)

        seg_x2 = UpsampleDeterministicP3D(2)(seg_x4)
        seg_x2 = seg_x2 + skip2
        seg_x2 = self.seg_x2(seg_x2)

        seg_x1 = UpsampleDeterministicP3D(2)(seg_x2)
        seg_x1 = seg_x1 + skip1
        seg_x1 = self.seg_x1(seg_x1)

        seg = self.seg_cls(seg_x1)
        # seg = seg_x1
        return seg

class U_CorResNet_Fix_prototype_FIM_IIM(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, weight_std=False):
        super(U_CorResNet_Fix_prototype_FIM_IIM, self).__init__()
        self.encoder = Unet_Encoder(in_channels, weight_std)
        self.decoder = Unet_Decoder(out_channels, weight_std)
        self.coarse_pred = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 2, kernel_size=1, stride=1),
            nn.Softmax(dim=1)
        )

    def getPrototype_CT(self, feature, mask):

        mask_ids = torch.unique(mask)
        mask_ids = mask_ids.long()
        prototype = []
        for i in mask_ids:
            temp_mask = (mask == i).int()
            temp_proto = torch.sum(feature * temp_mask, dim=(2, 3, 4)) \
                         / (temp_mask.sum(dim=(2, 3, 4)) + 1e-5)  # 1 x C
            prototype.append(temp_proto)
        prototype = torch.cat(prototype, dim=0)  # n*c
        prototype_CT = F.normalize(prototype, p=2, dim=1)
        return prototype_CT, mask_ids

    def getPrototype_batch(self, feature, mask):
        # feature = F.normalize(feature, p=2, dim=1)
        prototype_batch = []
        ptr_batch = []
        for i in range(feature.shape[0]):  # batch
            prototype = []
            this_feature = feature[i]  # C, D, H, W
            this_mask = mask[i]
            this_mask_ids = torch.unique(this_mask).long()
            for j in this_mask_ids:
                temp_mask = (this_mask == j).int()
                temp_proto = torch.sum(this_feature * temp_mask, dim=(1, 2, 3)) \
                             / (temp_mask.sum(dim=(1, 2, 3)) + 1e-5)  # 1 x C
                prototype.append(temp_proto.unsqueeze(0))
            prototype = torch.cat(prototype, dim=0)  # n*c
            prototype = F.normalize(prototype, p=2, dim=1)
            prototype_batch.append(prototype)
            ptr_batch.append(this_mask_ids)

        return prototype_batch, ptr_batch
    def data_encoder(self, inputs, mask, mode='original'):
        feature, skip1, skip2, skip3 = self.encoder(inputs)

        up_feature = UpsampleDeterministicP3D(8)(feature)
        if mode == 'aug':
            coarse = None
        else:
            coarse = self.coarse_pred(feature)
            coarse = UpsampleDeterministicP3D(8)(coarse)

        proto_CT, ptr = self.getPrototype_CT(up_feature, mask)
        return feature, skip1, skip2, skip3, proto_CT, ptr, coarse

    def data_decoder(self, feature, skip1, skip2, skip3):
        output = self.decoder(feature, skip1, skip2, skip3)

        return output

    def forward(self, inputs, mask):
        feature, skip1, skip2, skip3 = self.encoder(inputs)
        output = self.decoder(feature, skip1, skip2, skip3)
        return output

class U_CorResNet_Fix_BL(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, weight_std=False):
        super(U_CorResNet_Fix_BL, self).__init__()
        self.encoder = Unet_Encoder(in_channels, weight_std=weight_std)
        self.decoder = Unet_Decoder(out_channels, weight_std=weight_std)

    def forward(self, inputs):
        feature, skip1, skip2, skip3 = self.encoder(inputs)
        output = self.decoder(feature, skip1, skip2, skip3)
        return output, feature

#
if __name__ == "__main__":

    # device_ids = [1, 2, 3]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    net = U_CorResNet_Fix_BL()
    net = net.cuda()
    x = torch.ones(1, 1, 56, 144, 144).cuda()
    y1 = net(x)
    print(y1.size())
