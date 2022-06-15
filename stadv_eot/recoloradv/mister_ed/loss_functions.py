# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/mister_ed/loss_functions.py
#
# The license for the original version of this file can be
# found in the `recoloradv` directory (LICENSE_RECOLORADV).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import torch.nn as nn
import torch
from numbers import Number
from .utils import pytorch_utils as utils
from .utils import image_utils as img_utils
from . import spatial_transformers as st
from torch.autograd import Variable
from functools import partial
from . import adversarial_perturbations as ap

""" Loss function building blocks """


##############################################################################
#                                                                            #
#                        LOSS FUNCTION WRAPPER                               #
#                                                                            #
##############################################################################

class RegularizedLoss(object):
    """ Wrapper for multiple PartialLoss objects where we combine with
        regularization constants """

    def __init__(self, losses, scalars, negate=False):
        """
        ARGS:
            losses : dict - dictionary of partialLoss objects, each is keyed
                            with a nice identifying name
            scalars : dict - dictionary of scalars, each is keyed with the
                             same identifying name as is in self.losses
            negate : bool - if True, we negate the whole thing at the end
        """

        assert sorted(losses.keys()) == sorted(scalars.keys())

        self.losses = losses
        self.scalars = scalars
        self.negate = negate

    def forward(self, examples, labels, *args, **kwargs):

        output = None
        output_per_example = kwargs.get('output_per_example', False)
        for k in self.losses:
            loss = self.losses[k]
            scalar = self.scalars[k]

            loss_val = loss.forward(examples, labels, *args, **kwargs)
            # assert scalar is either a...
            assert (isinstance(scalar, float) or  # number
                    scalar.numel() == 1 or  # tf wrapping of a number
                    scalar.shape == loss_val.shape)  # same as the loss_val

            addendum = loss_val * scalar
            if addendum.numel() > 1:
                if not output_per_example:
                    addendum = torch.sum(addendum)

            if output is None:
                output = addendum
            else:
                output = output + addendum
        if self.negate:
            return output * -1
        else:
            return output

    def setup_attack_batch(self, fix_im):
        """ Setup before calling loss on a new minibatch. Ensures the correct
            fix_im for reference regularizers and that all grads are zeroed
        ARGS:
            fix_im: Variable (NxCxHxW) - Ground images for this minibatch
                    SHOULD BE IN [0.0, 1.0] RANGE
        """
        for loss in self.losses.values():
            if isinstance(loss, ReferenceRegularizer):
                loss.setup_attack_batch(fix_im)
            else:
                loss.zero_grad()

    def cleanup_attack_batch(self):
        """ Does some cleanup stuff after we finish on a minibatch:
        - clears the fixed images for ReferenceRegularizers
        - zeros grads
        - clears example-based scalars (i.e. scalars that depend on which
          example we're using)
        """
        for loss in self.losses.values():
            if isinstance(loss, ReferenceRegularizer):
                loss.cleanup_attack_batch()
            else:
                loss.zero_grad()

        for key, scalar in self.scalars.items():
            if not isinstance(scalar, Number):
                self.scalars[key] = None

    def zero_grad(self):
        for loss in self.losses.values():
            loss.zero_grad()  # probably zeros the same net more than once...


class PartialLoss(object):
    """ Partially applied loss object. Has forward and zero_grad methods """

    def __init__(self):
        self.nets = []

    def zero_grad(self):
        for net in self.nets:
            net.zero_grad()


##############################################################################
#                                                                            #
#                                  LOSS FUNCTIONS                            #
#                                                                            #
##############################################################################

############################################################################
#                       NAIVE CORRECT INDICATOR LOSS                       #
############################################################################

class IncorrectIndicator(PartialLoss):
    def __init__(self, classifier, normalizer=None):
        super(IncorrectIndicator, self).__init__()
        self.classifier = classifier
        self.normalizer = normalizer

    def forward(self, examples, labels, *args, **kwargs):
        """ Returns either (the number | a boolean vector) of examples that
            don't match the labels when run through the
            classifier(normalizer(.)) composition.
        ARGS:
            examples: Variable (NxCxHxW) - should be same shape as
                      ctx.fix_im, is the examples we define loss for.
                      SHOULD BE IN [0.0, 1.0] RANGE
            labels: Variable (longTensor of length N) - true classification
                    output for fix_im/examples
        KWARGS:
            return_type: String - either 'int' or 'vector'. If 'int', we return
                         the number of correctly classified examples,
                         if 'vector' we return a boolean length-N longtensor
                         with the indices of
        RETURNS:
            scalar loss variable or boolean vector, depending on kwargs
        """
        return_type = kwargs.get('return_type', 'int')
        assert return_type in ['int', 'vector']

        class_out = self.classifier.forward(self.normalizer.forward(examples))

        _, outputs = torch.max(class_out, 1)
        incorrect_indicator = outputs != labels

        if return_type == 'int':
            return torch.sum(incorrect_indicator)
        else:
            return incorrect_indicator


##############################################################################
#                                   Standard XEntropy Loss                   #
##############################################################################

class PartialXentropy(PartialLoss):
    def __init__(self, classifier, normalizer=None):
        super(PartialXentropy, self).__init__()
        self.classifier = classifier
        self.normalizer = normalizer
        self.nets.append(self.classifier)

    def forward(self, examples, labels, *args, **kwargs):
        """ Returns XEntropy loss
        ARGS:
            examples: Variable (NxCxHxW) - should be same shape as
                      ctx.fix_im, is the examples we define loss for.
                      SHOULD BE IN [0.0, 1.0] RANGE
            labels: Variable (longTensor of length N) - true classification
                    output for fix_im/examples
        RETURNS:
            scalar loss variable
        """

        if self.normalizer is not None:
            normed_examples = self.normalizer.forward(examples)
        else:
            normed_examples = examples

        xentropy_init_kwargs = {}
        if kwargs.get('output_per_example') == True:
            xentropy_init_kwargs['reduction'] = 'none'
        criterion = nn.CrossEntropyLoss(**xentropy_init_kwargs)
        return criterion(self.classifier.forward(normed_examples), labels)


##############################################################################
#                           Carlini Wagner loss functions                    #
##############################################################################

class CWLossF6(PartialLoss):
    def __init__(self, classifier, normalizer=None, kappa=0.0):
        super(CWLossF6, self).__init__()
        self.classifier = classifier
        self.normalizer = normalizer
        self.nets.append(self.classifier)
        self.kappa = kappa

    def forward(self, examples, labels, *args, **kwargs):
        classifier_in = self.normalizer.forward(examples)
        classifier_out = self.classifier.forward(classifier_in)

        # get target logits
        target_logits = torch.gather(classifier_out, 1, labels.view(-1, 1))

        # get largest non-target logits
        max_2_logits, argmax_2_logits = torch.topk(classifier_out, 2, dim=1)
        top_max, second_max = max_2_logits.chunk(2, dim=1)
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
        targets_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        targets_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_other = targets_eq_max * second_max + targets_ne_max * top_max

        if kwargs.get('targeted', False):
            # in targeted case, want to make target most likely
            f6 = torch.clamp(max_other - target_logits, min=-1 * self.kappa)
        else:
            # in NONtargeted case, want to make NONtarget most likely
            f6 = torch.clamp(target_logits - max_other, min=-1 * self.kappa)

        return f6.squeeze()


##############################################################################
#                                                                            #
#                               REFERENCE REGULARIZERS                       #
#                                                                            #
##############################################################################
""" Regularization terms that refer back to a set of 'fixed images', or the
    original images.
    example: L2 regularization which computes L2dist between a perturbed image
             and the FIXED ORIGINAL IMAGE

    NOTE: it's important that these return Variables that are scalars
    (output.numel() == 1), otherwise there's a memory leak w/ CUDA.
    See my discussion on this here:
        https://discuss.pytorch.org/t/cuda-memory-not-being-freed/15965
"""


class ReferenceRegularizer(PartialLoss):
    def __init__(self, fix_im):
        super(ReferenceRegularizer, self).__init__()
        self.fix_im = fix_im

    def setup_attack_batch(self, fix_im):
        """ Setup function to ensure fixed images are set
            has been made; also zeros grads
        ARGS:
            fix_im: Variable (NxCxHxW) - Ground images for this minibatch
                    SHOULD BE IN [0.0, 1.0] RANGE
        """
        self.fix_im = fix_im
        self.zero_grad()

    def cleanup_attack_batch(self):
        """ Cleanup function to clear the fixed images after an attack batch
            has been made; also zeros grads
        """
        old_fix_im = self.fix_im
        self.fix_im = None
        del old_fix_im
        self.zero_grad()


#############################################################################
#                               SOFT L_INF REGULARIZATION                   #
#############################################################################

class SoftLInfRegularization(ReferenceRegularizer):
    '''
        see page 10 of this paper (https://arxiv.org/pdf/1608.04644.pdf)
        for discussion on why we want SOFT l inf
    '''

    def __init__(self, fix_im, **kwargs):
        super(SoftLInfRegularization, self).__init__(fix_im)

    def forward(self, examples, *args, **kwargs):
        # ARGS should have one element, which serves as the tau value

        tau = 8.0 / 255.0  # starts at 1 each time?
        scale_factor = 0.9
        l_inf_dist = float(torch.max(torch.abs(examples - self.fix_im)))
        '''
        while scale_factor * tau > l_inf_dist:
            tau *= scale_factor

        assert tau > l_inf_dist
        '''
        delta_minus_taus = torch.clamp(torch.abs(examples - self.fix_im) - tau,
                                       min=0.0)
        batchwise = utils.batchwise_norm(delta_minus_taus, 'inf', dim=0)
        return batchwise.squeeze()


#############################################################################
#                               L2 REGULARIZATION                           #
#############################################################################

class L2Regularization(ReferenceRegularizer):

    def __init__(self, fix_im, **kwargs):
        super(L2Regularization, self).__init__(fix_im)

    def forward(self, examples, *args, **kwargs):
        l2_dist = img_utils.nchw_l2(examples, self.fix_im,
                                    squared=True).view(-1, 1)
        return l2_dist.squeeze()


#############################################################################
#                         LPIPS PERCEPTUAL REGULARIZATION                   #
#############################################################################

class LpipsRegularization(ReferenceRegularizer):

    def __init__(self, fix_im, **kwargs):
        super(LpipsRegularization, self).__init__(fix_im)

        manual_gpu = kwargs.get('manual_gpu', None)
        if manual_gpu is not None:
            self.use_gpu = manual_gpu
        else:
            self.use_gpu = utils.use_gpu()

        self.dist_model = dm.DistModel(net='alex', manual_gpu=self.use_gpu)

    def forward(self, examples, *args, **kwargs):
        xform = lambda im: im * 2.0 - 1.0
        perceptual_loss = self.dist_model.forward_var(examples,
                                                      self.fix_im)

        return perceptual_loss.squeeze()


#############################################################################
#                         SSIM PERCEPTUAL REGULARIZATION                    #
#############################################################################

class SSIMRegularization(ReferenceRegularizer):

    def __init__(self, fix_im, **kwargs):
        super(SSIMRegularization, self).__init__(fix_im)

        if 'window_size' in kwargs:
            self.ssim_instance = ssim.SSIM(window_size=kwargs['window_size'])
        else:
            self.ssim_instance = ssim.SSIM()

        manual_gpu = kwargs.get('manual_gpu', None)
        if manual_gpu is not None:
            self.use_gpu = manual_gpu
        else:
            self.use_gpu = utils.use_gpu()

    def forward(self, examples, *args, **kwargs):
        output = []
        for ex, fix_ex in zip(examples, self.fix_im):
            output.append(1.0 - self.ssim_instance(ex.unsqueeze(0),
                                                   fix_ex.unsqueeze(0)))
        return torch.stack(output)


##############################################################################
#                                                                            #
#                           SPATIAL LOSS FUNCTIONS                           #
#                                                                            #
##############################################################################

class FullSpatialLpLoss(PartialLoss):
    """ Spatial loss using lp norms on the spatial transformation parameters
    This is defined as the Lp difference between the identity map and the
    provided spatial transformation parameters
    """

    def __init__(self, **kwargs):
        super(FullSpatialLpLoss, self).__init__()

        lp = kwargs.get('lp', 2)
        assert lp in [1, 2, 'inf']
        self.lp = lp

    def forward(self, examples, *args, **kwargs):
        """ Computes lp loss between identity map and spatial transformation.
            There better be a kwarg with key 'spatial' which is as FullSpatial
            object describing how the examples were generated from the originals
        """
        st_obj = kwargs['spatial']
        assert isinstance(st_obj, st.FullSpatial)

        # First create the identity map and make same type as examples
        identity_map = Variable(st_obj.identity_params(examples.shape))
        if examples.is_cuda:
            identity_map.cuda()

        # Then take diffs and take lp norms
        diffs = st_obj.grid_params - identity_map
        lp_norm = utils.batchwise_norm(diffs, self.lp, dim=0)
        return lp_norm  # return Nx1 variable, will sum in parent class


class PerturbationNormLoss(PartialLoss):

    def __init__(self, **kwargs):
        super(PerturbationNormLoss, self).__init__()

        lp = kwargs.get('lp', 2)
        assert lp in [1, 2, 'inf']
        self.lp = lp

    def forward(self, examples, *args, **kwargs):
        """ Computes perturbation norm and multiplies by scale
        There better be a kwarg with key 'perturbation' which is a perturbation
        object with a 'perturbation_norm' method that takes 'lp_style' as a
        kwarg
        """

        perturbation = kwargs['perturbation']
        assert isinstance(perturbation, ap.AdversarialPerturbation)

        return perturbation.perturbation_norm(lp_style=self.lp)


##############################################################################
#                                                                            #
#                       Combined Transformer Loss                            #
#                                                                            #
##############################################################################

class CombinedTransformerLoss(ReferenceRegularizer):
    """ General class for distance functions and loss functions of the form
    min_T ||X - T(Y)|| + c * || T ||
    where X is the original image, and Y is the 'adversarial' input image.
    """

    def __init__(self, fix_im, transform_class=None,
                 regularization_constant=1.0,
                 transformation_loss=partial(utils.summed_lp_norm, lp=2),
                 transform_norm_kwargs=None):
        """ Takes in a reference fix im and a class of transformations we need
            to search over to compute forward.
        """
        super(CombinedTransformerLoss, self).__init__(fix_im)
        self.transform_class = transform_class
        self.regularization_constant = regularization_constant
        self.transformation_loss = transformation_loss
        self.transform_norm_kwargs = transform_norm_kwargs or {}
        self.transformer = None

    def cleanup_attack_batch(self):
        super(CombinedTransformerLoss, self).cleanup_attack_batch()
        self.transformer = None

    def _inner_loss(self, examples):
        """ Computes the combined loss for a particular transformation """

        trans_examples = self.transformer.forward(examples)
        trans_loss = self.transformation_loss(self.fix_im - trans_examples)

        trans_norm = self.transformer.norm(**self.transform_norm_kwargs)
        return trans_loss + trans_norm * self.regularization_constant

    def forward(self, examples, *args, **kwargs):
        """ Computes the distance between examples and args
        ARGS:
            examples : NxCxHxW Variable - 'adversarially' perturbed image from
                       the self.fix_im
        KWARGS:
            optimization stuff here
        """

        ######################################################################
        #   Setup transformer + optimizer                                    #
        ######################################################################
        self.transformer = self.transform_class(shape=examples.shape)

        optim_kwargs = kwargs.get('xform_loss_optim_kwargs', {})
        optim_type = kwargs.get('xform_loss_optim_type', torch.optim.Adam)
        num_iter = kwargs.get('xform_loss_num_iter', 20)

        optimizer = optim_type(self.transformer.parameters(), **optim_kwargs)

        #####################################################################
        #   Iterate and optimize the transformer                            #
        #####################################################################
        for iter_no in range(num_iter):
            optimizer.zero_grad()
            loss = self._inner_loss(examples)
            loss.backward()
            optimizer.step()

        return self._inner_loss(examples)


class RelaxedTransformerLoss(ReferenceRegularizer):
    """  Relaxed version of transformer loss: assumes that the adversarial
         examples are of the form Y=S(X) + delta for some S in the
         transformation class and some small delta perturbation outside the
         perturbation.

         In this case, we just compute ||delta|| + c||S||

         This saves us from having to do the inner minmization step
    """

    def __init__(self, fix_im,
                 regularization_constant=1.0,
                 transformation_loss=partial(utils.summed_lp_norm, lp=2),
                 transform_norm_kwargs=None):
        """ Takes in a reference fix im and a class of transformations we need
            to search over to compute forward.
        """
        super(RelaxedTransformerLoss, self).__init__(fix_im)
        self.regularization_constant = regularization_constant
        self.transformation_loss = transformation_loss
        self.transform_norm_kwargs = transform_norm_kwargs or {}

    def forward(self, examples, *args, **kwargs):
        """ Computes the distance between examples and args
        ARGS:
            examples : NxCxHxW Variable - 'adversarially' perturbed image from
                       the self.fix_im
        KWARGS:
            optimization stuff here
        """

        # Collect transformer norm
        transformer = kwargs['transformer']
        assert isinstance(transformer, st.ParameterizedTransformation)

        transformer_norm = self.regularization_constant * \
                           transformer.norm(**self.transform_norm_kwargs)

        # Collect transformation loss
        delta = self.transformer.forward(self.fix_im) - examples
        transformation_loss = self.transformation_loss(delta)

        return transformation_loss + transformer_norm
