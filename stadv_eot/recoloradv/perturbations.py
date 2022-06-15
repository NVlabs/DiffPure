# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/perturbations.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_RECOLORADV).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

from .mister_ed import adversarial_perturbations as ap
from .mister_ed.adversarial_perturbations import initialized
from .mister_ed.utils import pytorch_utils as utils

from . import color_transformers as ct
from . import color_spaces as cs


class ReColorAdv(ap.AdversarialPerturbation):
    """
    Puts the color at each pixel in the image through the same transformation.

    Parameters:
     - lp_style: number or 'inf'
     - lp_bound: maximum norm of color transformation. Can be a tensor of size
       (num_channels,), in which case each channel will be bounded by the
       cooresponding bound in the tensor. For instance, passing
       [0.1, 0.15, 0.05] would allow a norm of 0.1 for R, 0.15 for G, and 0.05
       for B. Not supported by all transformations.
     - use_smooth_loss: whether to optimize using the loss function
       for FullSpatial that rewards smooth vector fields
     - xform_class: a subclass of
       color_transformers.ParameterizedTransformation
     - xform_params: dict of parameters to pass to the xform_class.
     - cspace_class: a subclass of color_spaces.ColorSpace that indicates
       in which color space the transformation should be performed
       (RGB by default)
    """

    def __init__(self, threat_model, perturbation_params, *other_args):
        super().__init__(threat_model, perturbation_params)
        assert issubclass(perturbation_params.xform_class,
                          ct.ParameterizedTransformation)

        self.lp_style = perturbation_params.lp_style
        self.lp_bound = perturbation_params.lp_bound
        self.use_smooth_loss = perturbation_params.use_smooth_loss
        self.scalar_step = perturbation_params.scalar_step or 1.0
        self.cspace = perturbation_params.cspace or cs.RGBColorSpace()

    def _merge_setup(self, num_examples, new_xform):
        """ DANGEROUS TO BE CALLED OUTSIDE OF THIS FILE!!!"""
        self.num_examples = num_examples
        self.xform = new_xform
        self.initialized = True

    def setup(self, originals):
        super().setup(originals)
        self.xform = self.perturbation_params.xform_class(
            shape=originals.shape, manual_gpu=self.use_gpu,
            cspace=self.cspace,
            **(self.perturbation_params.xform_params or {}),
        )
        self.initialized = True

    @initialized
    def perturbation_norm(self, x=None, lp_style=None):
        lp_style = lp_style or self.lp_style
        if self.use_smooth_loss:
            assert isinstance(self.xform, ct.FullSpatial)
            return self.xform.smoothness_norm()
        else:
            return self.xform.norm(lp=lp_style)

    @initialized
    def constrain_params(self, x=None):
        # Do lp projections
        if isinstance(self.lp_style, int) or self.lp_style == 'inf':
            self.xform.project_params(self.lp_style, self.lp_bound)

    @initialized
    def update_params(self, step_fxn):
        param_list = list(self.xform.parameters())
        assert len(param_list) == 1
        params = param_list[0]
        assert params.grad.data is not None
        self.add_to_params(step_fxn(params.grad.data) * self.scalar_step)

    @initialized
    def add_to_params(self, grad_data):
        """ Assumes only one parameters object in the Spatial Transform """
        param_list = list(self.xform.parameters())
        assert len(param_list) == 1
        params = param_list[0]
        params.data.add_(grad_data)

    @initialized
    def random_init(self):
        param_list = list(self.xform.parameters())
        assert len(param_list) == 1
        param = param_list[0]
        random_perturb = utils.random_from_lp_ball(param.data,
                                                   self.lp_style,
                                                   self.lp_bound)

        param.data.add_(self.xform.identity_params +
                        random_perturb - self.xform.xform_params.data)

    @initialized
    def merge_perturbation(self, other, self_mask):
        super().merge_perturbation(other, self_mask)
        new_perturbation = ReColorAdv(self.threat_model,
                                      self.perturbation_params)

        new_xform = self.xform.merge_xform(other.xform, self_mask)
        new_perturbation._merge_setup(self.num_examples, new_xform)

        return new_perturbation

    def forward(self, x):
        if not self.initialized:
            self.setup(x)
        self.constrain_params()

        return self.cspace.to_rgb(
            self.xform.forward(self.cspace.from_rgb(x)))
