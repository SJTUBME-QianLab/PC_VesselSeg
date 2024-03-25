#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Time    : 2021/5/5 10:46
# @Author  : Dirk Li
# @Email   : junli.dirk@gmail.com
# @File    : transform.py
# @Software: PyCharm,Windows10
# @Hardware: 32G-RAM,Intel-i7-7700k,GTX-1080

import numpy as np
from batchgenerators.transforms import Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    SpatialTransform_2,
    MirrorTransform,
)


def get_train_transform(patch_size):
    # we now create a list of transforms. These are not necessarily the best transforms to use for BraTS, this is just
    # to showcase some things
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    tr_transforms.append(
        SpatialTransform_2(
            patch_size=None,
            do_elastic_deform=True,
            deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
            angle_y=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
            angle_z=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
            do_scale=True,
            scale=(0.75, 1.25),
            border_mode_data="constant",
            border_cval_data=0,
            border_mode_seg="constant",
            border_cval_seg=0,
            order_seg=1,
            order_data=3,
            random_crop=True,
            p_el_per_sample=0.1,
            p_rot_per_sample=0.1,
            p_scale_per_sample=0.1,
        )
    )

    # now we mirror along all axes
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness transform for 15% of samples
    tr_transforms.append(
        BrightnessMultiplicativeTransform(
            (0.7, 1.5), per_channel=True, p_per_sample=0.15
        )
    )

    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    tr_transforms.append(
        GammaTransform(
            gamma_range=(0.5, 2),
            invert_image=False,
            per_channel=True,
            p_per_sample=0.15,
        )
    )
    # we can also invert the image, apply the transform and then invert back
    tr_transforms.append(
        GammaTransform(
            gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15
        )
    )

    # Gaussian Noise
    tr_transforms.append(
        GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15)
    )

    # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
    # thus make the model more robust to it
    tr_transforms.append(
        GaussianBlurTransform(
            blur_sigma=(0.5, 1.5),
            different_sigma_per_channel=True,
            p_per_channel=0.5,
            p_per_sample=0.15,
        )
    )

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms
