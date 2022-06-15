# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/color_transformers.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_RECOLORADV).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

"""
Contains various parameterizations for spatial transformation in 3D color space.
"""

import torch
import torch.nn as nn
from .mister_ed.utils import pytorch_utils as utils
from torch.autograd import Variable
from . import norms
from functools import lru_cache


##############################################################################
#                                                                            #
#                               SKELETON CLASS                               #
#                                                                            #
##############################################################################

class ParameterizedTransformation(nn.Module):
    """ General class of transformations.
    All subclasses need the following methods:
    - norm: no args -> scalar variable
    - identity_params: shape -> TENSOR : takes an input shape and outputs
                       the subclass-specific parameter for the identity
                       transformation
    - forward : Variable -> Variable - is the transformation
    """

    def __init__(self, **kwargs):
        super(ParameterizedTransformation, self).__init__()

        if kwargs.get('manual_gpu', None) is not None:
            self.use_gpu = kwargs['manual_gpu']
        else:
            self.use_gpu = utils.use_gpu()

    def clone(self, shape=None, example_index=None):
        raise NotImplementedError()

    def norm(self, lp='inf'):
        raise NotImplementedError("Need to call subclass's norm!")

    @classmethod
    def identity_params(self, shape):
        raise NotImplementedError("Need to call subclass's identity_params!")

    def merge_xform(self, other, self_mask):
        """ Takes in an other instance of this same class with the same
            shape of parameters (NxSHAPE) and a self_mask bytetensor of length
            N and outputs the merge between self's parameters for the indices
            of 1s in the self_mask and other's parameters for the indices of 0's
        ARGS:
            other: instance of same class as self with params of shape NxSHAPE -
                   the thing we merge with this one
            self_mask : ByteTensor (length N) - which indices of parameters we
                        keep from self, and which we keep from other
        RETURNS:
            New instance of this class that's merged between the self and other
            (same shaped params)
        """

        # JUST DO ASSERTS IN THE SKELETON CLASS
        assert self.__class__ == other.__class__

        self_params = self.xform_params.data
        other_params = other.xform_params.data
        assert self_params.shape == other_params.shape
        assert self_params.shape[0] == self_mask.shape[0]
        assert other_params.shape[0] == self_mask.shape[0]

        new_xform = self.__class__(shape=self.img_shape)

        new_params = utils.fold_mask(self.xform_params.data,
                                     other.xform_params.data, self_mask)
        new_xform.xform_params = nn.Parameter(new_params)
        new_xform.use_gpu = self.use_gpu
        return new_xform

    def forward(self, examples):
        raise NotImplementedError("Need to call subclass's forward!")


class AffineTransform(ParameterizedTransformation):
    def __init__(self, *args, **kwargs):
        super(AffineTransform, self).__init__(**kwargs)
        img_shape = kwargs['shape']
        self.img_shape = img_shape
        self.xform_params = nn.Parameter(self.identity_params(img_shape))

    def clone(self, shape=None, example_index=None):
        xform = AffineTransform(shape=shape or self.img_shape)
        if example_index is None:
            my_params = self.xform_params
        else:
            my_params = self.xform_params[example_index][None]
        xform.xform_params = nn.Parameter(
            my_params.clone()
                .expand(shape[0], -1, -1)
        )
        return xform

    def norm(self, lp='inf'):
        identity_params = Variable(self.identity_params(self.img_shape))
        return utils.batchwise_norm(self.xform_params - identity_params, lp,
                                    dim=0)

    def identity_params(self, shape):
        num_examples = shape[0]
        identity_affine_transform = torch.zeros(num_examples, 3, 4)
        if self.use_gpu:
            identity_affine_transform = identity_affine_transform.cuda()

        identity_affine_transform[:, 0, 0] = 1
        identity_affine_transform[:, 1, 1] = 1
        identity_affine_transform[:, 2, 2] = 1

        return identity_affine_transform

    def project_params(self, lp, lp_bound):
        assert isinstance(lp, int) or lp == 'inf'
        diff = self.xform_params.data - self.identity_params(self.img_shape)
        new_diff = utils.batchwise_lp_project(diff, lp, lp_bound)
        self.xform_params.data.add_(new_diff - diff)

    def forward(self, x):
        N, _, W, H = self.img_shape
        x_padded = torch.cat([x, torch.ones(N, 1, W, H)], 1).permute(
            0, 2, 3, 1)
        transform_padded = self.xform_params[:, None, None, :, :] \
            .expand(-1, W, H, -1, -1)
        x_transformed = transform_padded.matmul(x_padded[..., None]) \
            .squeeze(4) \
            .permute(0, 3, 1, 2)
        return x_transformed


class FullSpatial(ParameterizedTransformation):
    def __init__(self, *args, resolution_x=8,
                 resolution_y=8, resolution_z=8, **kwargs):
        super(FullSpatial, self).__init__(**kwargs)

        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z

        img_shape = kwargs['shape']
        self.img_shape = img_shape

        self.cspace = kwargs.get('cspace')

        batch_size = self.img_shape[0]
        self.identity_params = FullSpatial.construct_identity_params(
            batch_size,
            self.resolution_x,
            self.resolution_y,
            self.resolution_z,
            torch.cuda.current_device() if self.use_gpu else None,
        )
        self.xform_params = nn.Parameter(
            torch.empty_like(self.identity_params)
                .copy_(self.identity_params)
        )

    def clone(self, shape=None, example_index=None):
        xform = FullSpatial(
            shape=shape or self.img_shape,
            resolution_x=self.resolution_x,
            resolution_y=self.resolution_y,
            resolution_z=self.resolution_z,
            cspace=self.cspace,
        )
        if example_index is None:
            my_params = self.xform_params
        else:
            my_params = self.xform_params[example_index][None]
        xform.xform_params = nn.Parameter(
            my_params.clone()
                .expand(shape[0], -1, -1, -1, -1)
        )
        return xform

    def smoothness_norm(self):
        return norms.smoothness(self.xform_params -
                                self.identity_params)

    def norm(self, lp='inf'):
        if isinstance(lp, int) or lp == 'inf':
            return utils.batchwise_norm(
                self.xform_params - self.identity_params,
                lp, dim=0,
            )
        else:
            assert lp == 'smooth'
            return self.smoothness_norm()

    def clip_params(self):
        """
        Clips the parameters to be between 0 and 1 and also within the color
        space's gamut.
        """

        clamp_params = torch.clamp(self.xform_params, 0, 1).data

        params_shape = self.xform_params.size()
        flattened_params = (
            clamp_params
                .permute(0, 4, 1, 2, 3)
                .reshape(params_shape[0], 3, -1, 1)
        )
        gamut_params = self.cspace.from_rgb(self.cspace.to_rgb(
            flattened_params))
        clamp_params = (
            gamut_params
                .permute(0, 2, 3, 1)
                .reshape(*params_shape)
        )

        change_in_params = clamp_params - self.xform_params.data
        self.xform_params.data.add_(change_in_params)

    def merge_xform(self, other, self_mask):
        """
        Takes in an other instance of this same class with the same
        shape of parameters (NxSHAPE) and a self_mask bytetensor of length
        N and outputs the merge between self's parameters for the indices
        of 1s in the self_mask and other's parameters for the indices of 0's
        """

        super().merge_xform(other, self_mask)
        new_xform = FullSpatial(shape=self.img_shape,
                                manual_gpu=self.use_gpu,
                                resolution_x=self.resolution_x,
                                resolution_y=self.resolution_y,
                                resolution_z=self.resolution_z,
                                cspace=self.cspace)
        new_params = utils.fold_mask(self.xform_params.data,
                                     other.xform_params.data, self_mask)
        new_xform.xform_params = nn.Parameter(new_params)

        return new_xform

    def project_params(self, lp, lp_bound):
        """
        Projects the params to be within lp_bound (according to an lp)
        of the identity map. First thing we do is clip the params to be
        valid, too.
        ARGS:
            lp : int or 'inf' - which LP norm we use. Must be an int or the
                 string 'inf'.
            lp_bound : float - how far we're allowed to go in LP land. Can be
                 a list to indicate that we can go more in some channels
                 than others.
        RETURNS:
            None, but modifies self.xform_params
        """

        assert isinstance(lp, int) or lp == 'inf'

        # clip first
        self.clip_params()

        # then project back
        if lp == 'inf':
            try:
                # first, assume lp_bound is a vector, and then revert to scalar
                # if it's not
                clamped_channels = []
                for channel_index, bound in enumerate(lp_bound):
                    clamped_channels.append(utils.clamp_ref(
                        self.xform_params[..., channel_index],
                        self.identity_params[..., channel_index],
                        bound,
                    ))
                clamp_params = torch.stack(clamped_channels, 4)
            except TypeError:
                clamp_params = utils.clamp_ref(self.xform_params.data,
                                               self.identity_params, lp_bound)
            change_in_params = clamp_params - self.xform_params.data
        else:
            flattened_params = (
                    self.xform_params.data -
                    self.identity_params
            ).reshape((-1, 3))
            projected_params = flattened_params.renorm(lp, 0, lp_bound)
            flattened_change = projected_params - flattened_params
            change_in_params = flattened_change.reshape(
                self.xform_params.size())
        self.xform_params.data.add_(change_in_params)

    def forward(self, imgs):
        device = torch.device('cuda') if self.use_gpu else None
        N, C, W, H = self.img_shape
        imgs = imgs.permute(0, 2, 3, 1)  # N x W x H x C
        imgs = imgs * torch.tensor(
            [
                self.resolution_x - 1,
                self.resolution_y - 1,
                self.resolution_z - 1,
            ],
            dtype=torch.float,
            device=device,
        )[None, None, None, :].expand(N, W, H, C)
        integer_part, float_part = torch.floor(imgs).long(), imgs % 1
        params_list = self.xform_params.view(N, -1, 3)

        # do trilinear interpolation from the params grid
        endpoint_values = []
        for delta_x in [0, 1]:
            corner_values = []
            for delta_y in [0, 1]:
                vertex_values = []
                for delta_z in [0, 1]:
                    params_index = Variable(torch.zeros(
                        N, W, H,
                        dtype=torch.long,
                        device=device,
                    ))
                    for color_index, resolution in [
                        (integer_part[..., 0] + delta_x, self.resolution_x),
                        (integer_part[..., 1] + delta_y, self.resolution_y),
                        (integer_part[..., 2] + delta_z, self.resolution_z),
                    ]:
                        color_index = color_index.clamp(
                            0, resolution - 1)
                        params_index = (params_index * resolution +
                                        color_index)
                    params_index = params_index.view(N, -1)[:, :, None] \
                        .expand(-1, -1, 3)
                    vertex_values.append(
                        params_list.gather(1, params_index)
                            .view(N, W, H, C)
                    )
                corner_values.append(
                    vertex_values[0] * (1 - float_part[..., 2, None]) +
                    vertex_values[1] * float_part[..., 2, None]
                )
            endpoint_values.append(
                corner_values[0] * (1 - float_part[..., 1, None]) +
                corner_values[1] * float_part[..., 1, None]
            )
        result = (
                endpoint_values[0] * (1 - float_part[..., 0, None]) +
                endpoint_values[1] * float_part[..., 0, None]
        )
        return result.permute(0, 3, 1, 2)

    @staticmethod
    @lru_cache(maxsize=10)
    def construct_identity_params(batch_size, resolution_x, resolution_y,
                                  resolution_z, device):
        identity_params = torch.empty(
            batch_size, resolution_x, resolution_y,
            resolution_z, 3,
            dtype=torch.float,
            device=device,
        )
        for x in range(resolution_x):
            for y in range(resolution_y):
                for z in range(resolution_z):
                    identity_params[:, x, y, z, 0] = \
                        x / (resolution_x - 1)
                    identity_params[:, x, y, z, 1] = \
                        y / (resolution_y - 1)
                    identity_params[:, x, y, z, 2] = \
                        z / (resolution_z - 1)
        return identity_params
