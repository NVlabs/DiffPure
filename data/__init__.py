# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from .datasets import imagenet_lmdb_dataset, imagenet_lmdb_dataset_sub, cifar10_dataset_sub

def get_transform(dataset_name, transform_type, base_size=256):
    from . import datasets
    if dataset_name == 'celebahq':
        return datasets.get_transform(dataset_name, transform_type, base_size)
    elif 'imagenet' in dataset_name:
        return datasets.get_transform(dataset_name, transform_type, base_size)
    else:
        raise NotImplementedError


def get_dataset(dataset_name, partition, *args, **kwargs):
    from . import datasets
    if dataset_name == 'celebahq':
        return datasets.CelebAHQDataset(partition, *args, **kwargs)
    else:
        raise NotImplementedError