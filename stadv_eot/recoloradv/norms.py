# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/norms.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_RECOLORADV).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import torch
from torch.autograd import Variable


def smoothness(grid):
    """
    Given a variable of dimensions (N, X, Y, [Z], C), computes the sum of
    the differences between adjacent points in the grid formed by the
    dimensions X, Y, and (optionally) Z. Returns a tensor of dimension N.
    """

    num_dims = len(grid.size()) - 2
    batch_size = grid.size()[0]
    norm = Variable(torch.zeros(batch_size, dtype=grid.data.dtype,
                                device=grid.data.device))

    for dim in range(num_dims):
        slice_before = (slice(None),) * (dim + 1)
        slice_after = (slice(None),) * (num_dims - dim)
        shifted_grids = [
            # left
            torch.cat([
                grid[slice_before + (slice(1, None),) + slice_after],
                grid[slice_before + (slice(-1, None),) + slice_after],
            ], dim + 1),
            # right
            torch.cat([
                grid[slice_before + (slice(None, 1),) + slice_after],
                grid[slice_before + (slice(None, -1),) + slice_after],
            ], dim + 1)
        ]
        for shifted_grid in shifted_grids:
            delta = shifted_grid - grid
            norm_components = (delta.pow(2).sum(-1) + 1e-10).pow(0.5)
            norm.add_(norm_components.sum(
                tuple(range(1, len(norm_components.size())))))

    return norm
