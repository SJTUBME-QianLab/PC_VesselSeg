#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Time    : 2020/8/12 15:03
# @Author  : Dirk Li
# @Email   : junli.dirk@gmail.com
# @File    : Upsample.py
# @Software: PyCharm,Windows10
# @Hardware: 32G-RAM,Intel-i7-7700k,GTX-1080

from torch import nn


def upsample_deterministic(x, upscale):
    """
    x: 4-dim tensor. shape is (batch,channel,h,w)
    output: 4-dim tensor. shape is (batch,channel,self. upscale*h,self. upscale*w)
    """
    return (
        x[:, :, :, None, :, None]
        .expand(-1, -1, -1, upscale, -1, upscale)
        .reshape(x.size(0), x.size(1), x.size(2) * upscale, x.size(3) * upscale)
    )


class UpsampleDeterministic(nn.Module):
    def __init__(self, upscale=2):
        """
        Upsampling in pytorch is not deterministic (at least for the version of 1.0.1)
        see https://github.com/pytorch/pytorch/issues/12207
        """
        super(UpsampleDeterministic, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        """
        x: 4-dim tensor. shape is (batch,channel,h,w)
        output: 4-dim tensor. shape is (batch,channel,upscale*h,upscale*w)
        """
        return upsample_deterministic(x, self.upscale)


def upsample_deterministicP3D(x, upscale):
    """
    x: 4-dim tensor. shape is (batch,channel,h,w)
    output: 4-dim tensor. shape is (batch,channel,self. upscale*h,self. upscale*w)
    """
    return (
        x[:, :, :, None, :, None, :, None]
        .expand(-1, -1, -1, upscale, -1, upscale, -1, upscale)
        .reshape(
            x.size(0),
            x.size(1),
            x.size(2) * upscale,
            x.size(3) * upscale,
            x.size(4) * upscale,
        )
    )


class UpsampleDeterministicP3D(nn.Module):
    def __init__(self, upscale=2):
        """
        Upsampling in pytorch is not deterministic (at least for the version of 1.0.1)
        see https://github.com/pytorch/pytorch/issues/12207
        """
        super(UpsampleDeterministicP3D, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        """
        x: 4-dim tensor. shape is (batch,channel,h,w)
        output: 4-dim tensor. shape is (batch,channel,upscale*h,upscale*w)
        """
        return upsample_deterministicP3D(x, self.upscale)


def upsample_deterministicP3D_without_Z(x, upscale):
    """
    x: 4-dim tensor. shape is (batch,channel,h,w)
    output: 4-dim tensor. shape is (batch,channel,self. upscale*h,self. upscale*w)
    """
    return (
        x[:, :, :, :, None, :, None]
        .expand(-1, -1, -1, -1, upscale, -1, upscale)
        .reshape(
            x.size(0), x.size(1), x.size(2), x.size(3) * upscale, x.size(4) * upscale
        )
    )


class UpsampleDeterministicP3D_without_Z(nn.Module):
    def __init__(self, upscale=2):
        """
        Upsampling in pytorch is not deterministic (at least for the version of 1.0.1)
        see https://github.com/pytorch/pytorch/issues/12207
        """
        super(UpsampleDeterministicP3D_without_Z, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        """
        x: 4-dim tensor. shape is (batch,channel,h,w)
        output: 4-dim tensor. shape is (batch,channel,upscale*h,upscale*w)
        """
        return upsample_deterministicP3D_without_Z(x, self.upscale)
