import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.conv_blocks import Conv3x3
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=stride, padding=1)
        self.bn2 = nn.InstanceNorm3d(out_channels)

        if in_channels != out_channels:
            self.res_conv = nn.Conv3d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            res = self.res_conv(x)
        else:
            res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        out = x + res
        out = self.relu(out)

        return out

class SegSEBlock(nn.Module):
    def __init__(self, in_channels, rate=2):
        super(SegSEBlock, self).__init__()
        conv = nn.Conv3d
        self.in_channels = in_channels
        self.rate = rate
        self.dila_conv = conv(self.in_channels, self.in_channels // self.rate, 3, padding=2, dilation=self.rate)
        self.conv1 = conv(self.in_channels // self.rate, self.in_channels, 1)

    def forward(self, input):
        x = self.dila_conv(input)
        x = self.conv1(x)
        x = nn.Sigmoid()(x)

        return x

class RecombinationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normalization=True, kernel_size=3):
        super(RecombinationBlock, self).__init__()
        conv = nn.Conv3d
        bn = nn.InstanceNorm3d

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bach_normalization = batch_normalization
        self.kerenl_size = kernel_size
        self.rate = 2
        self.expan_channels = self.out_channels * self.rate

        self.expansion_conv = conv(self.in_channels, self.expan_channels, 1)
        self.skip_conv = conv(self.in_channels, self.out_channels, 1)
        self.zoom_conv = conv(self.out_channels * self.rate, self.out_channels, 1)

        self.bn = bn(self.expan_channels)
        self.norm_conv = conv(self.expan_channels, self.expan_channels, self.kerenl_size, padding=1)

        self.segse_block = SegSEBlock(self.expan_channels)

    def forward(self, input):
        x = self.expansion_conv(input)

        for i in range(1):
            if self.bach_normalization:
                x = self.bn(x)
            x = nn.ReLU6()(x)
            x = self.norm_conv(x)

        se_x = self.segse_block(x)

        x = x * se_x

        x = self.zoom_conv(x)

        skip_x = self.skip_conv(input)
        out = x + skip_x

        return out

class AGC_block(nn.Module):
    # Attention-Guided Concatenation module
    def __init__(self, in_channels, out_channels):
        super(AGC_block, self).__init__()

        self.in1 = nn.Sequential(nn.Conv3d(in_channels, in_channels, 3, padding=1),
                                 nn.BatchNorm3d(in_channels),
                                 nn.ReLU())
        self.in2 = nn.Sequential(nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                 nn.BatchNorm3d(out_channels),
                                 nn.ReLU())

        self.up = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2),
                                nn.BatchNorm3d(out_channels),
                                nn.ReLU())
        self.sa = SA(out_channels)
        self.conv = nn.Sequential(nn.Conv3d(out_channels*2, out_channels, 1),
                                 nn.BatchNorm3d(out_channels),
                                 nn.ReLU())
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, fh, fl):
        fh = self.in1(fh)
        fl = self.in2(fl)

        fh = self.up(fh)
        map = self.sa(fh)
        fl = torch.mul(map, fl)

        f = torch.cat([fh, fl], 1)
        f = self.conv(f)
        f = self.bn(f)
        f = self.relu(f)
        return f


class SA(nn.Module):
    def __init__(self, in_channels):
        super(SA, self).__init__()
        middle_channel = int(in_channels/16)
        self.conv1 = nn.Conv3d(in_channels, middle_channel, kernel_size=1)
        self.bn = nn.BatchNorm3d(middle_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(middle_channel, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

class MSFF(nn.Module):
    def __init__(self, in_channels):
        super(MSFF, self).__init__()
        self.in1 = nn.Sequential(nn.Conv3d(in_channels, in_channels, 1),
                                 nn.BatchNorm3d(in_channels),
                                 nn.ReLU())
        self.in_channels = in_channels
        channel = int(in_channels/4)

        self.conv2 = Conv3x3(channel, channel, mode='3D')
        self.conv3 = Conv3x3(channel, channel, mode='3D')
        self.conv4 = Conv3x3(channel, channel, mode='3D')
        self.conv = nn.Conv3d(channel, channel, 1)
        self.out = nn.Sequential(nn.Conv3d(in_channels, in_channels, 1),
                                 nn.BatchNorm3d(in_channels),
                                 nn.ReLU())

    def forward(self, x):
        x = self.in1(x)

        # split
        flt = int(self.in_channels/4)
        x1 = x[:, 0:flt, :, :, :]
        x2 = x[:, flt:flt*2, :, :, :]
        x3 = x[:, flt*2:flt*3, :, :, :]
        x4 = x[:, flt*3:flt*4, :, :, :]

        x1 = x1
        x2 = self.conv2(x2)
        x3 = self.conv3(x2+x3)
        x4 = self.conv4(x3+x4)

        xx = x1+x2+x3+x4
        xx = self.conv(xx)

        x1 = x1 + xx
        x2 = x2 + xx
        x3 = x3 + xx
        x4 = x4 + xx

        xxx = torch.cat([x1, x2, x3, x4], 1)
        out = self.out(xxx)
        return out








