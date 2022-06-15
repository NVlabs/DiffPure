# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/color_spaces.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_RECOLORADV).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

"""
Contains classes that convert from RGB to various other color spaces and back.
"""

import torch
import numpy as np
import math


class ColorSpace(object):
    """
    Base class for color spaces.
    """

    def from_rgb(self, imgs):
        """
        Converts an Nx3xWxH tensor in RGB color space to a Nx3xWxH tensor in
        this color space. All outputs should be in the 0-1 range.
        """
        raise NotImplementedError()

    def to_rgb(self, imgs):
        """
        Converts an Nx3xWxH tensor in this color space to a Nx3xWxH tensor in
        RGB color space.
        """
        raise NotImplementedError()


class RGBColorSpace(ColorSpace):
    """
    RGB color space. Just applies identity transformation.
    """

    def from_rgb(self, imgs):
        return imgs

    def to_rgb(self, imgs):
        return imgs


class YPbPrColorSpace(ColorSpace):
    """
    YPbPr color space. Uses ITU-R BT.601 standard by default.
    """

    def __init__(self, kr=0.299, kg=0.587, kb=0.114, luma_factor=1,
                 chroma_factor=1):
        self.kr, self.kg, self.kb = kr, kg, kb
        self.luma_factor = luma_factor
        self.chroma_factor = chroma_factor

    def from_rgb(self, imgs):
        r, g, b = imgs.permute(1, 0, 2, 3)

        y = r * self.kr + g * self.kg + b * self.kb
        pb = (b - y) / (2 * (1 - self.kb))
        pr = (r - y) / (2 * (1 - self.kr))

        return torch.stack([y * self.luma_factor,
                            pb * self.chroma_factor + 0.5,
                            pr * self.chroma_factor + 0.5], 1)

    def to_rgb(self, imgs):
        y_prime, pb_prime, pr_prime = imgs.permute(1, 0, 2, 3)
        y = y_prime / self.luma_factor
        pb = (pb_prime - 0.5) / self.chroma_factor
        pr = (pr_prime - 0.5) / self.chroma_factor

        b = pb * 2 * (1 - self.kb) + y
        r = pr * 2 * (1 - self.kr) + y
        g = (y - r * self.kr - b * self.kb) / self.kg

        return torch.stack([r, g, b], 1).clamp(0, 1)


class ApproxHSVColorSpace(ColorSpace):
    """
    Converts from RGB to approximately the HSV cone using a much smoother
    transformation.
    """

    def from_rgb(self, imgs):
        r, g, b = imgs.permute(1, 0, 2, 3)

        x = r * np.sqrt(2) / 3 - g / (np.sqrt(2) * 3) - b / (np.sqrt(2) * 3)
        y = g / np.sqrt(6) - b / np.sqrt(6)
        z, _ = imgs.max(1)

        return torch.stack([z, x + 0.5, y + 0.5], 1)

    def to_rgb(self, imgs):
        z, xp, yp = imgs.permute(1, 0, 2, 3)
        x, y = xp - 0.5, yp - 0.5

        rp = float(np.sqrt(2)) * x
        gp = -x / np.sqrt(2) + y * np.sqrt(3 / 2)
        bp = -x / np.sqrt(2) - y * np.sqrt(3 / 2)

        delta = z - torch.max(torch.stack([rp, gp, bp], 1), 1)[0]
        r, g, b = rp + delta, gp + delta, bp + delta

        return torch.stack([r, g, b], 1).clamp(0, 1)


class HSVConeColorSpace(ColorSpace):
    """
    Converts from RGB to the HSV "cone", where (x, y, z) =
    (s * v cos h, s * v sin h, v). Note that this cone is then squashed to fit
    in [0, 1]^3 by letting (x', y', z') = ((x + 1) / 2, (y + 1) / 2, z).

    WARNING: has a very complex derivative, not very useful in practice
    """

    def from_rgb(self, imgs):
        r, g, b = imgs.permute(1, 0, 2, 3)

        mx, argmx = imgs.max(1)
        mn, _ = imgs.min(1)
        chroma = mx - mn
        eps = 1e-10
        h_max_r = math.pi / 3 * (g - b) / (chroma + eps)
        h_max_g = math.pi / 3 * (b - r) / (chroma + eps) + math.pi * 2 / 3
        h_max_b = math.pi / 3 * (r - g) / (chroma + eps) + math.pi * 4 / 3

        h = (((argmx == 0) & (chroma != 0)).float() * h_max_r
             + ((argmx == 1) & (chroma != 0)).float() * h_max_g
             + ((argmx == 2) & (chroma != 0)).float() * h_max_b)

        x = torch.cos(h) * chroma
        y = torch.sin(h) * chroma
        z = mx

        return torch.stack([(x + 1) / 2, (y + 1) / 2, z], 1)

    def _to_rgb_part(self, h, chroma, v, n):
        """
        Implements the function f(n) defined here:
        https://en.wikipedia.org/wiki/HSL_and_HSV#Alternative_HSV_to_RGB
        """

        k = (n + h * math.pi / 3) % 6
        return v - chroma * torch.min(k, 4 - k).clamp(0, 1)

    def to_rgb(self, imgs):
        xp, yp, z = imgs.permute(1, 0, 2, 3)
        x, y = xp * 2 - 1, yp * 2 - 1

        # prevent NaN gradients when calculating atan2
        x_nonzero = (1 - 2 * (torch.sign(x) == -1).float()) * (torch.abs(x) + 1e-10)
        h = torch.atan2(y, x_nonzero)
        v = z.clamp(0, 1)
        chroma = torch.min(torch.sqrt(x ** 2 + y ** 2 + 1e-10), v)

        r = self._to_rgb_part(h, chroma, v, 5)
        g = self._to_rgb_part(h, chroma, v, 3)
        b = self._to_rgb_part(h, chroma, v, 1)

        return torch.stack([r, g, b], 1).clamp(0, 1)


class CIEXYZColorSpace(ColorSpace):
    """
    The 1931 CIE XYZ color space (assuming input is in sRGB).

    Warning: may have values outside [0, 1] range. Should only be used in
    the process of converting to/from other color spaces.
    """

    def from_rgb(self, imgs):
        # apply gamma correction
        small_values_mask = (imgs < 0.04045).float()
        imgs_corrected = (
                (imgs / 12.92) * small_values_mask +
                ((imgs + 0.055) / 1.055) ** 2.4 * (1 - small_values_mask)
        )

        # linear transformation to XYZ
        r, g, b = imgs_corrected.permute(1, 0, 2, 3)
        x = 0.4124 * r + 0.3576 * g + 0.1805 * b
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        z = 0.0193 * r + 0.1192 * g + 0.9504 * b

        return torch.stack([x, y, z], 1)

    def to_rgb(self, imgs):
        # linear transformation
        x, y, z = imgs.permute(1, 0, 2, 3)
        r = 3.2406 * x - 1.5372 * y - 0.4986 * z
        g = -0.9689 * x + 1.8758 * y + 0.0415 * z
        b = 0.0557 * x - 0.2040 * y + 1.0570 * z

        imgs = torch.stack([r, g, b], 1)

        # apply gamma correction
        small_values_mask = (imgs < 0.0031308).float()
        imgs_clamped = imgs.clamp(min=1e-10)  # prevent NaN gradients
        imgs_corrected = (
                (12.92 * imgs) * small_values_mask +
                (1.055 * imgs_clamped ** (1 / 2.4) - 0.055) *
                (1 - small_values_mask)
        )

        return imgs_corrected


class CIELUVColorSpace(ColorSpace):
    """
    Converts to the 1976 CIE L*u*v* color space.
    """

    def __init__(self, up_white=0.1978, vp_white=0.4683, y_white=1,
                 eps=1e-10):
        self.xyz_cspace = CIEXYZColorSpace()
        self.up_white = up_white
        self.vp_white = vp_white
        self.y_white = y_white
        self.eps = eps

    def from_rgb(self, imgs):
        x, y, z = self.xyz_cspace.from_rgb(imgs).permute(1, 0, 2, 3)

        # calculate u' and v'
        denom = x + 15 * y + 3 * z + self.eps
        up = 4 * x / denom
        vp = 9 * y / denom

        # calculate L*, u*, and v*
        small_values_mask = (y / self.y_white < (6 / 29) ** 3).float()
        y_clamped = y.clamp(min=self.eps)  # prevent NaN gradients
        L = (
                ((29 / 3) ** 3 * y / self.y_white) * small_values_mask +
                (116 * (y_clamped / self.y_white) ** (1 / 3) - 16) *
                (1 - small_values_mask)
        )
        u = 13 * L * (up - self.up_white)
        v = 13 * L * (vp - self.vp_white)

        return torch.stack([L / 100, (u + 100) / 200, (v + 100) / 200], 1)

    def to_rgb(self, imgs):
        L = imgs[:, 0, :, :] * 100
        u = imgs[:, 1, :, :] * 200 - 100
        v = imgs[:, 2, :, :] * 200 - 100

        up = u / (13 * L + self.eps) + self.up_white
        vp = v / (13 * L + self.eps) + self.vp_white

        small_values_mask = (L <= 8).float()
        y = (
                (self.y_white * L * (3 / 29) ** 3) * small_values_mask +
                (self.y_white * ((L + 16) / 116) ** 3) * (1 - small_values_mask)
        )
        denom = 4 * vp + self.eps
        x = y * 9 * up / denom
        z = y * (12 - 3 * up - 20 * vp) / denom

        return self.xyz_cspace.to_rgb(
            torch.stack([x, y, z], 1).clamp(0, 1.1)).clamp(0, 1)
