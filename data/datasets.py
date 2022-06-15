# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os, sys

import io
import lmdb

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset, Subset

import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import folder, ImageFolder


# ---------------------------------------------------------------------------------------------------

def remove_prefix(s, prefix):
    if s.startswith(prefix):
        s = s[len(prefix):]
    return s


class ImageDataset(VisionDataset):
    """
    modified from: https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    uses cached directory listing if available rather than walking directory
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader=folder.default_loader,
                 extensions=folder.IMG_EXTENSIONS, transform=None,
                 target_transform=None, is_valid_file=None, return_path=False):
        super(ImageDataset, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        cache = self.root.rstrip('/') + '.txt'
        if os.path.isfile(cache):
            print("Using directory list at: %s" % cache)
            with open(cache) as f:
                samples = []
                for line in f:
                    (path, idx) = line.strip().split(';')
                    samples.append((os.path.join(self.root, path), int(idx)))
        else:
            print("Walking directory: %s" % self.root)
            samples = folder.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
            with open(cache, 'w') as f:
                for line in samples:
                    path, label = line
                    f.write('%s;%d\n' % (remove_prefix(path, self.root).lstrip('/'), label))

        if len(samples) == 0:
            raise (RuntimeError(
                "Found 0 files in subfolders of: " + self.root + "\nSupported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.return_path = return_path

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_path:
            return sample, target, path
        return sample, target

    def __len__(self):
        return len(self.samples)


# ---------------------------------------------------------------------------------------------------

# get the attributes from celebahq subset
def make_table(root):
    filenames = sorted(os.listdir(f'{root}/images'))
    # filter out non-png files, rename it to jpg to match entries in list_attr_celeba.txt
    celebahq = [os.path.basename(f).replace('png', 'jpg')
                if f.endswith('png') else os.path.basename(f) for f in filenames]
    attr_gt = pd.read_csv(f'{root}/list_attr_celeba.txt',
                          skiprows=1, delim_whitespace=True, index_col=0)
    attr_celebahq = attr_gt.reindex(index=celebahq).replace(-1, 0)

    # get the train/test/val partitions
    partitions = {}
    with open(f'{root}/list_eval_partition.txt') as f:
        for line in f:
            filename, part = line.strip().split(' ')
            partitions[filename] = int(part)
    partitions_list = [partitions[fname] for fname in attr_celebahq.index]

    attr_celebahq['partition'] = partitions_list
    return attr_celebahq


###### dataset functions ######

class CelebAHQDataset(Dataset):
    def __init__(self, partition, attribute, root=None, fraction=None, data_seed=1,
                 chunk_length=None, chunk_idx=-1, **kwargs):
        if root is None:
            root = './dataset/celebahq'
        self.fraction = fraction
        self.dset = ImageDataset(root, **kwargs)

        # make table
        attr_celebahq = make_table(root)

        # convert from train/val/test to partition numbers
        part_to_int = dict(train=0, val=1, test=2)

        def get_partition_indices(part):
            return np.where(attr_celebahq['partition'] == part_to_int[part])[0]

        partition_idx = get_partition_indices(partition)

        # if we want to further subsample the dataset, just subsample
        # partition_idx and Subset() once
        if fraction is not None:
            print("Using a fraction of the original dataset")
            print("The original dataset has length %d" % len(partition_idx))
            new_length = int(fraction / 100 * len(partition_idx))
            rng = np.random.RandomState(data_seed)
            new_indices = rng.choice(partition_idx, new_length, replace=False)
            partition_idx = new_indices
            print("The subsetted dataset has length %d" % len(partition_idx))

        elif chunk_length is not None and chunk_idx > 0:
            print(f"Using a fraction of the original dataset with chunk_length: {chunk_length}, chunk_idx: {chunk_idx}")
            print("The original dataset has length %d" % len(partition_idx))
            new_indices = partition_idx[chunk_length * chunk_idx: chunk_length * (chunk_idx + 1)]
            partition_idx = new_indices
            print("The subsetted dataset has length %d" % len(partition_idx))

        self.dset = Subset(self.dset, partition_idx)
        attr_subset = attr_celebahq.iloc[partition_idx]
        self.attr_subset = attr_subset[attribute]
        print('attribute freq: %0.4f (%d / %d)' % (self.attr_subset.mean(),
                                                   self.attr_subset.sum(),
                                                   len(self.attr_subset)))

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        data = self.dset[idx]
        # first element is the class, replace it
        label = self.attr_subset[idx]
        return (data[0], label, *data[2:])


###### transformation functions ######

def get_transform(dataset, transform_type, base_size=256):
    if dataset.lower() == "celebahq":
        assert base_size == 256, base_size

        if transform_type == 'imtrain':
            return transforms.Compose([
                transforms.Resize(base_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif transform_type == 'imval':
            return transforms.Compose([
                transforms.Resize(base_size),
                # no horizontal flip for standard validation
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif transform_type == 'imcolor':
            return transforms.Compose([
                transforms.Resize(base_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=.05, contrast=.05,
                                       saturation=.05, hue=.05),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif transform_type == 'imcrop':
            return transforms.Compose([
                # 1024 + 32, or 256 + 8
                transforms.Resize(int(1.03125 * base_size)),
                transforms.RandomCrop(base_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif transform_type == 'tensorbase':
            # dummy transform for compatibility with other datasets
            return transforms.Lambda(lambda x: x)
        else:
            raise NotImplementedError

    elif "imagenet" in dataset.lower():
        assert base_size == 224, base_size

        if transform_type == 'imtrain':
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(base_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif transform_type == 'imval':
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(base_size),
                # no horizontal flip for standard validation
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError


################################################################################
# ImageNet - LMDB
###############################################################################

def lmdb_loader(path, lmdb_data):
    # In-memory binary streams
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode('ascii'))
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')


def imagenet_lmdb_dataset(
        root, transform=None, target_transform=None,
        loader=lmdb_loader):
    """
    You can create this dataloader using:
    train_data = imagenet_lmdb_dataset(traindir, transform=train_transform)
    valid_data = imagenet_lmdb_dataset(validdir, transform=val_transform)
    """

    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        root + '_faster_imagefolder.lmdb.pt')
    lmdb_path = os.path.join(
        root + '_faster_imagefolder.lmdb')
    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        print('Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = ImageFolder(
            root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        print('Saving pt to {}'.format(pt_path))
        print('Building lmdb to {}'.format(lmdb_path))
        env = lmdb.open(lmdb_path, map_size=1e12)
        with env.begin(write=True) as txn:
            for path, class_index in data_set.imgs:
                with open(path, 'rb') as f:
                    data = f.read()
                txn.put(path.encode('ascii'), data)
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)

    return data_set


def imagenet_lmdb_dataset_sub(
        root, transform=None, target_transform=None,
        loader=lmdb_loader, num_sub=-1, data_seed=0):
    data_set = imagenet_lmdb_dataset(
        root, transform=transform, target_transform=target_transform,
        loader=loader)

    if num_sub > 0:
        partition_idx = np.random.RandomState(data_seed).choice(len(data_set), num_sub, replace=False)
        data_set = Subset(data_set, partition_idx)

    return data_set


################################################################################
# CIFAR-10
###############################################################################

def cifar10_dataset_sub(root, transform=None, num_sub=-1, data_seed=0):
    val_data = torchvision.datasets.CIFAR10(root=root, transform=transform, download=True, train=False)

    if num_sub > 0:
        partition_idx = np.random.RandomState(data_seed).choice(len(val_data), num_sub, replace=False)
        val_data = Subset(val_data, partition_idx)

    return val_data
