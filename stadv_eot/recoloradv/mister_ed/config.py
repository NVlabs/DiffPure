# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/mister_ed/config.py
#
# The license for the original version of this file can be
# found in the `recoloradv` directory (LICENSE_RECOLORADV).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import os

config_dir = os.path.abspath(os.path.dirname(__file__))


def path_resolver(path):
    if path.startswith('~/'):
        return os.path.expanduser(path)

    if path.startswith('./'):
        return os.path.join(*[config_dir] + path.split('/')[1:])


DEFAULT_DATASETS_DIR = path_resolver('~/datasets')
MODEL_PATH = path_resolver('./pretrained_models/')
OUTPUT_IMAGE_PATH = path_resolver('./output_images/')

DEFAULT_BATCH_SIZE = 128
DEFAULT_WORKERS = 4
CIFAR10_MEANS = [0.485, 0.456, 0.406]
CIFAR10_STDS = [0.229, 0.224, 0.225]

WIDE_CIFAR10_MEANS = [0.4914, 0.4822, 0.4465]
WIDE_CIFAR10_STDS = [0.2023, 0.1994, 0.2010]

IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS = [0.229, 0.224, 0.225]
