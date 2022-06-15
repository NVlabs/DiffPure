# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import sys
import argparse
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from robustbench import load_model
import data


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def update_state_dict(state_dict, idx_start=9):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[idx_start:]  # remove 'module.0.' of dataparallel
        new_state_dict[name]=v

    return new_state_dict


# ------------------------------------------------------------------------
def get_accuracy(model, x_orig, y_orig, bs=64, device=torch.device('cuda:0')):
    n_batches = x_orig.shape[0] // bs
    acc = 0.
    for counter in range(n_batches):
        x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        output = model(x)
        acc += (output.max(1)[1] == y).float().sum()

    return (acc / x_orig.shape[0]).item()


def get_image_classifier(classifier_name):
    class _Wrapper_ResNet(nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.resnet = resnet
            self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(3, 1, 1)
            self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(3, 1, 1)

        def forward(self, x):
            x = (x - self.mu.to(x.device)) / self.sigma.to(x.device)
            return self.resnet(x)

    if 'imagenet' in classifier_name:
        if 'resnet18' in classifier_name:
            print('using imagenet resnet18...')
            model = models.resnet18(pretrained=True).eval()
        elif 'resnet50' in classifier_name:
            print('using imagenet resnet50...')
            model = models.resnet50(pretrained=True).eval()
        elif 'resnet101' in classifier_name:
            print('using imagenet resnet101...')
            model = models.resnet101(pretrained=True).eval()
        elif 'wideresnet-50-2' in classifier_name:
            print('using imagenet wideresnet-50-2...')
            model = models.wide_resnet50_2(pretrained=True).eval()
        elif 'deit-s' in classifier_name:
            print('using imagenet deit-s...')
            model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).eval()
        else:
            raise NotImplementedError(f'unknown {classifier_name}')

        wrapper_resnet = _Wrapper_ResNet(model)

    elif 'cifar10' in classifier_name:
        if 'wideresnet-28-10' in classifier_name:
            print('using cifar10 wideresnet-28-10...')
            model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-28-10-at0' in classifier_name:
            print('using cifar10 wrn-28-10-at0...')
            model = load_model(model_name='Gowal2021Improving_28_10_ddpm_100m', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-28-10-at1' in classifier_name:
            print('using cifar10 wrn-28-10-at1...')
            model = load_model(model_name='Gowal2020Uncovering_28_10_extra', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-70-16-at0' in classifier_name:
            print('using cifar10 wrn-70-16-at0...')
            model = load_model(model_name='Gowal2021Improving_70_16_ddpm_100m', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-70-16-at1' in classifier_name:
            print('using cifar10 wrn-70-16-at1...')
            model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-70-16-L2-at1' in classifier_name:
            print('using cifar10 wrn-70-16-L2-at1...')
            model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10',
                               threat_model='L2')  # pixel in [0, 1]

        elif 'wideresnet-70-16' in classifier_name:
            print('using cifar10 wideresnet-70-16 (dm_wrn-70-16)...')
            from robustbench.model_zoo.architectures.dm_wide_resnet import DMWideResNet, Swish
            model = DMWideResNet(num_classes=10, depth=70, width=16, activation_fn=Swish)  # pixel in [0, 1]

            model_path = 'pretrained/cifar10/wresnet-76-10/weights-best.pt'
            print(f"=> loading wideresnet-70-16 checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path)['model_state_dict']))
            model.eval()
            print(f"=> loaded wideresnet-70-16 checkpoint")

        elif 'resnet-50' in classifier_name:
            print('using cifar10 resnet-50...')
            from classifiers.cifar10_resnet import ResNet50
            model = ResNet50()  # pixel in [0, 1]

            model_path = 'pretrained/cifar10/resnet-50/weights.pt'
            print(f"=> loading resnet-50 checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            print(f"=> loaded resnet-50 checkpoint")

        elif 'wrn-70-16-dropout' in classifier_name:
            print('using cifar10 wrn-70-16-dropout (standard wrn-70-16-dropout)...')
            from classifiers.cifar10_resnet import WideResNet_70_16_dropout
            model = WideResNet_70_16_dropout()  # pixel in [0, 1]

            model_path = 'pretrained/cifar10/wrn-70-16-dropout/weights.pt'
            print(f"=> loading wrn-70-16-dropout checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            print(f"=> loaded wrn-70-16-dropout checkpoint")

        else:
            raise NotImplementedError(f'unknown {classifier_name}')

        wrapper_resnet = model

    elif 'celebahq' in classifier_name:
        attribute = classifier_name.split('__')[-1]  # `celebahq__Smiling`
        ckpt_path = f'pretrained/celebahq/{attribute}/net_best.pth'
        from classifiers.attribute_classifier import ClassifierWrapper
        model = ClassifierWrapper(attribute, ckpt_path=ckpt_path)
        wrapper_resnet = model
    else:
        raise NotImplementedError(f'unknown {classifier_name}')

    return wrapper_resnet


def load_data(args, adv_batch_size):
    if 'imagenet' in args.domain:
        val_dir = './dataset/imagenet_lmdb/val'  # using imagenet lmdb data
        val_transform = data.get_transform(args.domain, 'imval', base_size=224)
        val_data = data.imagenet_lmdb_dataset_sub(val_dir, transform=val_transform,
                                                  num_sub=args.num_sub, data_seed=args.data_seed)
        n_samples = len(val_data)
        val_loader = DataLoader(val_data, batch_size=n_samples, shuffle=False, pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(val_loader))
    elif 'cifar10' in args.domain:
        data_dir = './dataset'
        transform = transforms.Compose([transforms.ToTensor()])
        val_data = data.cifar10_dataset_sub(data_dir, transform=transform,
                                            num_sub=args.num_sub, data_seed=args.data_seed)
        n_samples = len(val_data)
        val_loader = DataLoader(val_data, batch_size=n_samples, shuffle=False, pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(val_loader))
    elif 'celebahq' in args.domain:
        data_dir = './dataset/celebahq'
        attribute = args.classifier_name.split('__')[-1]  # `celebahq__Smiling`
        val_transform = data.get_transform('celebahq', 'imval')
        clean_dset = data.get_dataset('celebahq', 'val', attribute, root=data_dir, transform=val_transform,
                                      fraction=2, data_seed=args.data_seed)  # data_seed randomizes here
        loader = DataLoader(clean_dset, batch_size=adv_batch_size, shuffle=False,
                            pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(loader))  # [0, 1], 256x256
    else:
        raise NotImplementedError(f'Unknown domain: {args.domain}!')

    print(f'x_val shape: {x_val.shape}')
    x_val, y_val = x_val.contiguous().requires_grad_(True), y_val.contiguous()
    print(f'x (min, max): ({x_val.min()}, {x_val.max()})')

    return x_val, y_val
