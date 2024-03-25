import os

import matplotlib.pyplot as plt

from config.config import (
    nih_data_path,
    nih_label_path,
    LOW_RANGE,
    HIGH_RANGE,
    H_NIH_3D,
    W_NIH_3D,
    msd_data_path,
    msd_multi_label_path,
    START_Z,
    END_Z,
)
from utils.util import padding_z, center_crop
from augmentation.volumentations import *


def get_augmentation():
    return Compose(
        [
            # RemoveEmptyBorder(always_apply=True),
            # RandomScale((0.8, 1.2)),
            # PadIfNeeded(patch_size, always_apply=True),
            # RandomCrop(patch_size, always_apply=True),
            # CenterCrop(patch_size, always_apply=True),
            # RandomCrop(patch_size, always_apply=True),
            # Resize(always_apply=True),
            # CropNonEmptyMaskIfExists(patch_size, always_apply=True),
            # Normalize(always_apply=True),
            ElasticTransform((0, 0.25)),
            RandomRotate((-15, 15), (-15, 15), (-15, 15)),
            RandomFlip(0),
            RandomFlip(1),
            RandomFlip(2),
            # Transpose((1,0,2)), # only if patch.height = patch.width
            # RandomRotate90((0,1)),
            RandomGamma(),
            RandomGaussianNoise(),
        ],
        p=1,
    )


if __name__ == "__main__":

    #############################################################################################

    # image = np.load(os.path.join(nih_data_path, '{:0>4}.npy'.format(1))).transpose(2, 0, 1).astype(
    #     np.float)
    # mask = np.load(os.path.join(nih_label_path, '{:0>4}.npy'.format(1))).transpose(2, 0, 1)
    #
    # np.minimum(np.maximum(image, LOW_RANGE, image), HIGH_RANGE, image)
    # image -= LOW_RANGE
    # image /= (HIGH_RANGE - LOW_RANGE)
    #
    # image, mask = padding_z(image, mask)
    # image, mask = center_crop(image, H_NIH_3D, W_NIH_3D), center_crop(mask, H_NIH_3D, W_NIH_3D)
    # img, lbl = image[::2, ::2, ::2].transpose(1, 2, 0), mask[::2, ::2, ::2].transpose(1, 2, 0)
    #
    # plt.figure()
    # plt.imshow(img[:, :, 60], 'gray')
    # plt.show()
    # plt.figure()
    # plt.imshow(lbl[:, :, 60], 'gray')
    # plt.show()
    #
    # patch_size = (lbl.shape[0], lbl.shape[1], lbl.shape[2])
    #
    # print(img.shape, lbl.shape)
    # # view_batch(img.transpose(2,0,1), lbl.transpose(2,0,1))
    #
    # aug = get_augmentation()
    #
    # data = {
    #     'image': img,
    #     'mask': lbl,
    #     'size': lbl.shape,
    # }
    #
    # aug_data = aug(**data)
    # img, lbl = aug_data['image'], aug_data['mask']
    # print(img.shape, lbl.shape, np.max(img), np.max(lbl))
    # # print(img)
    # plt.figure()
    # plt.imshow(img[:, :, 60], 'gray')
    # plt.show()
    # plt.figure()
    # plt.imshow(lbl[:, :, 60], 'gray')
    # plt.show()

    #############################################################################################

    image = np.load(os.path.join(msd_data_path, "{}.npy".format(1))).astype(np.float)
    mask = np.load(os.path.join(msd_multi_label_path, "{}.npy".format(1)))

    np.minimum(np.maximum(image, LOW_RANGE, image), HIGH_RANGE, image)
    image -= LOW_RANGE
    image /= HIGH_RANGE - LOW_RANGE

    slices = len(mask)
    aug = get_augmentation()

    if slices > 200:
        image, mask = (
            image[int(slices * START_Z) : int(slices * END_Z)],
            mask[int(slices * START_Z) : int(slices * END_Z)],
        )

    image, mask = padding_z(image, mask, mode="up")
    image, mask = (
        center_crop(image, H_NIH_3D, W_NIH_3D),
        center_crop(mask, H_NIH_3D, W_NIH_3D),
    )
    image, mask = image[:, ::2, ::2], mask[:, ::2, ::2]

    if True:
        temp_data = {
            "image": image.transpose(1, 2, 0),
            "mask": mask.transpose(1, 2, 0),
            "size": mask.shape,
        }

        aug_data = aug(**temp_data)
        img, lbl = (
            aug_data["image"].transpose(2, 0, 1),
            aug_data["mask"].transpose(2, 0, 1),
        )

    print(img.shape, lbl.shape, np.max(img), np.max(lbl))

    print(((lbl > 0) & (lbl < 1)).sum())
    print(((lbl > 1) & (lbl < 2)).sum())
    print(((lbl < 0)).sum())
    print(((lbl > 2)).sum())

    plt.figure()
    plt.imshow(img[60], "gray")
    plt.show()
    plt.figure()
    plt.imshow(lbl[60], "gray")
    plt.show()
