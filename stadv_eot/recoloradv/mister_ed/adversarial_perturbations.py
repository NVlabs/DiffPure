# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/mister_ed/adversarial_perturbations.py
#
# The license for the original version of this file can be
# found in the `recoloradv` directory (LICENSE_RECOLORADV).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

""" File that holds adversarial perturbations as torch.nn.Modules.
    An adversarial perturbation is an example-specific
"""

import torch
import torch.nn as nn
from . import spatial_transformers as st
from .utils import image_utils as img_utils
from .utils import pytorch_utils as utils
from torch.autograd import Variable
import functools


# assert initialized decorator
def initialized(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        assert self.initialized, ("Parameters not initialized yet. "
                                  "Call .forward(...) first")
        return func(self, *args, **kwargs)
    return wrapper

##############################################################################
#                                                                            #
#                                   SKELETON CLASS                           #
#                                                                            #
##############################################################################

class AdversarialPerturbation(nn.Module):
    """ Skeleton class to hold adversarial perturbations FOR A SINGLE MINIBATCH.
        For general input-agnostic adversarial perturbations, see the
        ThreatModel class

        All subclasses need the following:
        - perturbation_norm() : no args -> scalar Variable
        - self.parameters() needs to iterate over params we want to optimize
        - constrain_params() : no args -> no return,
             modifies the parameters such that this is still a valid image
        - forward : no args -> Variable - applies the adversarial perturbation
                    the originals and outputs a Variable of how we got there
        - adversarial_tensors() : applies the adversarial transform to the
                                  originals and outputs TENSORS that are the
                                  adversarial images
    """

    def __init__(self, threat_model, perturbation_params):

        super(AdversarialPerturbation, self).__init__()
        self.threat_model = threat_model
        self.initialized = False
        self.perturbation_params = perturbation_params

        if isinstance(perturbation_params, tuple):
            self.use_gpu = perturbation_params[1].use_gpu or utils.use_gpu()
        else:
            self.use_gpu = perturbation_params.use_gpu or utils.use_gpu()
        # Stores parameters of the adversarial perturbation and hyperparams
        # to compute total perturbation norm here


    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        if isinstance(self.perturbation_params, tuple):
            output_str = "[Perturbation] %s: %s" % (self.__class__.__name__,
                                                    self.perturbation_params[1])
            output_str += '\n['
            for el in self.perturbation_params[0]:
                output_str += '\n\t%s,' % el
            output_str += '\n]'
            return output_str
        else:
            return "[Perturbation] %s: %s"  % (self.__class__.__name__,
                                               self.perturbation_params)

    def _merge_setup(self, *args):
        """ Internal method to be used when initializing a new perturbation
            from merging only. Should not be called outside this file!!
        """
        pass

    def setup(self, x):
        """ This is the standard setup technique and should be used to
            initialize a perturbation (i.e. sets up parameters and unlocks
            all other methods)
        ARGS:
            x : Variable or Tensor (NxCxHxW) - the images this perturbation is
                intended for
        """
        self.num_examples = x.shape[0]


    @initialized
    def perturbation_norm(self, x=None):
        """ This returns the 'norm' of this perturbation. Optionally, for
            certain norms, having access to the images for which the
            perturbation is intended can have an effect on the output.
        ARGS:
            x : Variable or Tensor (NxCxHxW) - optionally can be the images
                that the perturbation was intended for
        RETURNS:
            Scalar Variable
        """
        raise NotImplementedError("Need to call subclass method here")

    @initialized
    def constrain_params(self):
        """ This modifies the parameters such that the perturbation falls within
            the threat model it belongs to. E.g. for l-infinity threat models,
            this clips the params to match the right l-infinity bound.

            TODO: for non-lp norms, projecting to the nearest point in the level
                  set
        """
        raise NotImplementedError("Need to call subclass method here")

    @initialized
    def make_valid_image(self, x):
        """ This takes in the minibatch self's parameters were tuned for and
            clips the parameters such that this is still a valid image.
        ARGS:
            x : Variable or Tensor (NxCxHxW) - the images this this perturbation
                was intended for
        RETURNS:
            None
        """
        pass # Only implement in classes that can create invalid images

    @initialized
    def forward(self, x):
        """ This takes in the minibatch self's parameters were tuned for and
            outputs a variable of the perturbation applied to the images
        ARGS:
            x : Variable (NxCxHxW) - the images this this perturbation
                was intended for
        RETURNS:
            Variable (NxCxHxW) - the perturbation applied to the input images
        """
        raise NotImplementedError("Need to call subclass method here")

    @initialized
    def add_to_params(self, grad_data):
        """ This takes in a Tensor the same shape as self's parameters and
            adds to them. Note that this usually won't preserve gradient
            information
            (also this might have different signatures in subclasses)
        ARGS:
            x : Tensor (params-shape) - Tensor to be added to the
                parameters of self
        RETURNS:
            None, but modifies self's parameters
        """
        raise NotImplementedError("Need to call subclass method here")

    @initialized
    def update_params(self, step_fxn):
        """ This takes in a function step_fxn: Tensor -> Tensor that generates
            the change to the parameters that we step along. This loops through
            all parameters and updates signs accordingly.
            For sequential perturbations, this also multiplies by a scalar if
            provided
        ARGS:
            step_fxn : Tensor -> Tensor - function that maps tensors to tensors.
                       e.g. for FGSM, we want a function that multiplies signs
                       by step_size
        RETURNS:
            None, but updates the parameters
        """
        raise NotImplementedError("Need to call subclass method here")


    @initialized
    def adversarial_tensors(self, x=None):
        """ Little helper method to get the tensors of the adversarial images
            directly
        """
        assert x is not None or self.originals is not None
        if x is None:
            x = self.originals

        return self.forward(x).data

    @initialized
    def attach_attr(self, attr_name, attr):
        """ Special method to set an attribute if it doesn't exist in this
            object yet. throws error if this attr already exists
        ARGS:
            attr_name : string - name of attribute we're attaching
            attr: object - attribute we're attaching
        RETURNS:
            None
        """
        if hasattr(self, attr_name):
            raise Exception("%s already has attribute %s" % (self, attr_name))
        else:
            setattr(self, attr_name, attr)


    @initialized
    def attach_originals(self, originals):
        """ Little helper method to tack on the original images to self to
            pass around the (images, perturbation) in a single object
        """
        self.attach_attr('originals', originals)


    @initialized
    def random_init(self):
        """ Modifies the parameters such that they're randomly initialized
            uniformly across the threat model (this is harder for nonLp threat
            models...). Takes no args and returns nothing, but modifies the
            parameters
        """
        raise NotImplementedError("Need to call subclass method here")

    @initialized
    def merge_perturbation(self, other, self_mask):
        """ Special technique to merge this perturbation with another
            perturbation of the same threat model.
            This will return a new perturbation object that, for each parameter
            will return the parameters of self for self_mask, and the
            perturbation of other for NOT(self_mask)

        ARGS:
            other: AdversarialPerturbation Object - instance of other
                   adversarial perturbation that is instantiated with the
                   same threat model as self
            self_indices: ByteTensor [N] : bytetensor indicating which
                          parameters to include from self and which to include
                          from other
        """

        # this parent class just does the shared asserts such that this is a
        # valid thing
        assert self.__class__ == other.__class__
        assert self.threat_model == other.threat_model
        assert self.num_examples == other.num_examples
        assert self.perturbation_params == other.perturbation_params
        assert other.initialized

    @initialized
    def collect_successful(self, classifier_net, normalizer):
        """ Returns a list of [adversarials, originals] of the SUCCESSFUL
            attacks only, according to the given classifier_net, normalizer
            SUCCESSFUL here means that the adversarial is different
        ARGS:
            TODO: fill in when I'm not in crunchtime
        """

        assert self.originals is not None
        adversarials = Variable(self.adversarial_tensors())
        originals = Variable(self.originals)

        adv_out = torch.max(classifier_net(normalizer(adversarials)), 1)[1]
        out = torch.max(classifier_net(normalizer(originals)), 1)[1]
        adv_idx_bytes = adv_out != out
        idxs = []
        for idx, el in enumerate(adv_idx_bytes):
            if float(el) > 0:
                idxs.append(idx)

        idxs = torch.LongTensor(idxs)
        if self.originals.is_cuda:
            idxs = idxs.cuda()

        return [torch.index_select(self.adversarial_tensors(), 0, idxs),
                torch.index_select(self.originals, 0, idxs)]

    @initialized
    def collect_adversarially_successful(self, classifier_net, normalizer,
                                         labels):
        """ Returns an object containing the SUCCESSFUL attacked examples,
            their corresponding originals, and the number of misclassified
            examples
        ARGS:
            classifier_net : nn.Module subclass - neural net that is the
                             relevant classifier
            normalizer : DifferentiableNormalize object - object to convert
                         input data to mean-zero, unit-var examples
            labels : Variable (longTensor N) - correct labels for classification
                     of self.originals
        RETURNS:
            dict with structure:
            {'adversarials': Variable(N'xCxHxW) - adversarial perturbation
                            applied
             'originals': Variable(N'xCxHxW) - unperturbed examples that
                                               were correctly classified AND
                                               successfully attacked
             'num_correctly_classified': int - number of correctly classified
                                               unperturbed examples
            }
        """
        assert self.originals is not None
        adversarials = Variable(self.adversarial_tensors())
        originals = Variable(self.originals)

        adv_out = torch.max(classifier_net(normalizer(adversarials)), 1)[1]
        out = torch.max(classifier_net(normalizer(originals)), 1)[1]

        # First take a subset of correctly classified originals
        correct_idxs = (out == labels) # correctly classified idxs
        adv_idx_bytes = (adv_out != out) # attacked examples

        num_correctly_classified = int(sum(correct_idxs))

        adv_idxs = adv_idx_bytes * correct_idxs


        idxs = []
        for idx, el in enumerate(adv_idxs):
            if float(el) > 0:
                idxs.append(idx)

        idxs = torch.LongTensor(idxs)
        if self.originals.is_cuda:
            idxs = idxs.cuda()


        return {'adversarial': torch.index_select(self.adversarial_tensors(),
                                                  0, idxs),
                'originals': torch.index_select(self.originals, 0, idxs),
                'num_correctly_classified': num_correctly_classified}



    @initialized
    def display(self, scale=5, successful_only=False, classifier_net=None,
                normalizer=None):
        """ Displays this adversarial perturbation in a 3-row format:
            top row is adversarial images, second row is original images,
            bottom row is difference magnified by scale (default 5)
        ARGS:
            scale: int - how much to magnify differences by
            successful_only: bool - if True we only display successful (in that
                             advs output different classifier outputs)
                             If this is not None, classifie_net and normalizer
                             cannot be None
        RETURNS:
            None, but displays images
        """
        if successful_only:
            assert classifier_net is not None
            assert normalizer is not None
            advs, origs = self.collect_successful(classifier_net, normalizer)
        else:
            advs = self.adversarial_tensors()
            origs = self.originals

        diffs = torch.clamp((advs - origs) * scale + 0.5, 0, 1)
        img_utils.show_images([advs, origs, diffs])


class PerturbationParameters(dict):
    """ Object that stores parameters like a dictionary.
        This allows perturbation classes to be only partially instantiated and
        then fed various 'originals' later.
    Implementation taken from : https://stackoverflow.com/a/14620633/3837607
    (and then modified with the getattribute trick to return none instead of
     error for missing attributes)
    """
    def __init__(self, *args, **kwargs):
        super(PerturbationParameters, self).__init__(*args, **kwargs)
        if kwargs.get('manual_gpu') is not None:
            self.use_gpu = kwargs['manual_gpu']
        else:
            self.use_gpu = utils.use_gpu()
        self.__dict__ = self

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return None


class ThreatModel(object):
    def __init__(self, perturbation_class, param_kwargs, *other_args):
        """ Factory class to generate per_minibatch instances of Adversarial
            perturbations.
        ARGS:
            perturbation_class : class - subclass of Adversarial Perturbations
            param_kwargs : dict - dict containing named kwargs to instantiate
                           the class in perturbation class
        """
        assert issubclass(perturbation_class, AdversarialPerturbation)
        self.perturbation_class = perturbation_class
        if isinstance(param_kwargs, dict):
            param_kwargs = PerturbationParameters(**param_kwargs)
        self.param_kwargs = param_kwargs
        self.other_args = other_args

    def __repr__(self):
        return "[Threat] %s: %s"  % (str(self.perturbation_class.__name__),
                                     self.param_kwargs)

    def __call__(self, *args):
        if args == ():
            return self.perturbation_obj()
        else:
            perturbation_obj = self.perturbation_obj()
            perturbation_obj.setup(*args)
            return perturbation_obj



    def perturbation_obj(self):
        return self.perturbation_class(self, self.param_kwargs, *self.other_args)



##############################################################################
#                                                                            #
#                            ADDITION PARAMETERS                             #
#                                                                            #
##############################################################################

class DeltaAddition(AdversarialPerturbation):

    def __init__(self, threat_model, perturbation_params, *other_args):
        """ Maintains a delta that gets addded to the originals to generate
            adversarial images. This is the type of adversarial perturbation
            that the literature extensivey studies
        ARGS:
            threat_model : ThreatModel object that is used to initialize self
            perturbation_params: PerturbationParameters object.
                { lp_style : None, int or 'inf' - if not None is the type of
                            Lp_bound that we apply to this adversarial example
                lp_bound : None or float - cannot be None if lp_style is
                           not None, but if not None should be the lp bound
                           we allow for adversarial perturbations
                custom_norm : None or fxn:(NxCxHxW) -> Scalar Variable. This is
                              not implemented for now
                }
        """

        super(DeltaAddition, self).__init__(threat_model, perturbation_params)
        self.lp_style = perturbation_params.lp_style
        self.lp_bound = perturbation_params.lp_bound
        if perturbation_params.custom_norm is not None:
            raise NotImplementedError("Only LP norms allowed for now")
        self.scalar_step = perturbation_params.scalar_step or 1.0


    def _merge_setup(self, num_examples, delta_data):
        """ DANGEROUS TO BE CALLED OUTSIDE OF THIS FILE!!!"""
        self.num_examples = num_examples
        self.delta = nn.Parameter(delta_data)
        self.initialized = True


    def setup(self, x):
        super(DeltaAddition, self).setup(x)
        self.delta = nn.Parameter(torch.zeros_like(x))
        self.initialized = True

    @initialized
    def perturbation_norm(self, x=None, lp_style=None):
        lp_style = lp_style or self.lp_style
        assert isinstance(lp_style, int) or lp_style == 'inf'
        return utils.batchwise_norm(self.delta, lp=lp_style)


    @initialized
    def constrain_params(self):
        new_delta = utils.batchwise_lp_project(self.delta.data, self.lp_style,
                                               self.lp_bound)
        delta_diff = new_delta - self.delta.data
        self.delta.data.add_(delta_diff)

    @initialized
    def make_valid_image(self, x):
        new_delta = self.delta.data
        change_in_delta = utils.clamp_0_1_delta(new_delta, x)
        self.delta.data.add_(change_in_delta)

    @initialized
    def update_params(self, step_fxn):
        assert self.delta.grad.data is not None
        self.add_to_params(step_fxn(self.delta.grad.data) * self.scalar_step)

    @initialized
    def add_to_params(self, grad_data):
        """ sets params to be self.params + grad_data """
        self.delta.data.add_(grad_data)


    @initialized
    def random_init(self):
        self.delta = nn.Parameter(utils.random_from_lp_ball(self.delta.data,
                                                            self.lp_style,
                                                            self.lp_bound))

    @initialized
    def merge_perturbation(self, other, self_mask):
        super(DeltaAddition, self).merge_perturbation(other, self_mask)

        # initialize a new perturbation
        new_perturbation = DeltaAddition(self.threat_model,
                                         self.perturbation_params)

        # make the new parameters
        new_delta = utils.fold_mask(self.delta.data, other.delta.data,
                                    self_mask)

        # do the merge setup and return the object
        new_perturbation._merge_setup(self.num_examples,
                                      new_delta)
        return new_perturbation


    def forward(self, x):
        if not self.initialized:
            self.setup(x)
        self.make_valid_image(x) # not sure which one to do first...
        self.constrain_params()
        return x + self.delta




##############################################################################
#                                                                            #
#                               SPATIAL PARAMETERS                           #
#                                                                            #
##############################################################################

class ParameterizedXformAdv(AdversarialPerturbation):

    def __init__(self, threat_model, perturbation_params, *other_args):
        super(ParameterizedXformAdv, self).__init__(threat_model,
                                                    perturbation_params)
        assert issubclass(perturbation_params.xform_class,
                          st.ParameterizedTransformation)

        self.lp_style = perturbation_params.lp_style
        self.lp_bound = perturbation_params.lp_bound
        self.use_stadv = perturbation_params.use_stadv
        self.scalar_step = perturbation_params.scalar_step or 1.0


    def _merge_setup(self, num_examples, new_xform):
        """ DANGEROUS TO BE CALLED OUTSIDE OF THIS FILE!!!"""
        self.num_examples = num_examples
        self.xform = new_xform
        self.initialized = True

    def setup(self, originals):
        super(ParameterizedXformAdv, self).setup(originals)
        self.xform = self.perturbation_params.xform_class(shape=originals.shape,
                                                        manual_gpu=self.use_gpu)
        self.initialized = True

    @initialized
    def perturbation_norm(self, x=None, lp_style=None):
        lp_style = lp_style or self.lp_style
        if self.use_stadv is not None:
            assert isinstance(self.xform, st.FullSpatial)
            return self.xform.stAdv_norm()
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

        param.data.add_(self.xform.identity_params(self.xform.img_shape) +
                        random_perturb - self.xform.xform_params.data)


    @initialized
    def merge_perturbation(self, other, self_mask):
        super(ParameterizedXformAdv, self).merge_perturbation(other, self_mask)
        new_perturbation = ParameterizedXformAdv(self.threat_model,
                                                 self.perturbation_params)

        new_xform = self.xform.merge_xform(other.xform, self_mask)
        new_perturbation._merge_setup(self.num_examples, new_xform)

        return new_perturbation


    def forward(self, x):
        if not self.initialized:
            self.setup(x)
        self.constrain_params()
        return self.xform.forward(x)




##############################################################################
#                                                                            #
#                            SPATIAL + ADDITION PARAMETERS                   #
#                                                                            #
##############################################################################

class SequentialPerturbation(AdversarialPerturbation):
    """ Takes a list of perturbations and composes them. A norm needs to
        be specified here to describe the perturbations.
    """

    def __init__(self, threat_model, perturbation_sequence,
                 global_parameters=PerturbationParameters(pad=10),
                 preinit_pipeline=None):
        """ Initializes a sequence of adversarial perturbation layers
        ARGS:
            originals : NxCxHxW tensor - original images we create adversarial
                        perturbations for
            perturbation_sequence : ThreatModel[]  -
                list of ThreatModel objects
            global_parameters : PerturbationParameters - global parameters to
                                use. These contain things like how to norm this
                                sequence, how to constrain this sequence, etc
            preinit_pipelines: list[]
                if not None i
         """
        super(SequentialPerturbation, self).__init__(threat_model,
                                                    (perturbation_sequence,
                                                     global_parameters))

        if preinit_pipeline is not None:
            layers = preinit_pipeline
        else:
            layers = []
            for threat_model in perturbation_sequence:
                assert isinstance(threat_model, ThreatModel)
                layers.append(threat_model())

        self.pipeline = []
        for layer_no, layer in enumerate(layers):
            self.pipeline.append(layer)
            self.add_module('layer_%02d' % layer_no, layer)


        # norm: pipeline -> Scalar Variable
        self.norm = global_parameters.norm
        self.norm_weights = global_parameters.norm_weights

        # padding with black is useful to not throw information away during
        # sequential steps
        self.pad = nn.ConstantPad2d(global_parameters.pad or 0, 0)
        self.unpad = nn.ConstantPad2d(-1 * (global_parameters.pad or 0), 0)




    def _merge_setup(self, num_examples):
        self.num_examples = num_examples
        self.initialized = True


    def setup(self, x):
        super(SequentialPerturbation, self).setup(x)
        x = self.pad(x)
        for layer in self.pipeline:
            layer.setup(x)
        self.initialized = True


    @initialized
    def perturbation_norm(self, x=None, lp_style=None):
        # Need to define a nice way to describe the norm here. This can be
        # an empirical norm between input/output
        # For now, let's just say it's the sum of the norms of each constituent
        if self.norm is not None:
            return self.norm(self.pipeline, x=x, lp_style=lp_style)
        else:
            norm_weights = self.norm_weights or\
                              [1.0 for _ in range(len(self.pipeline))]
            out = None
            for i, layer in enumerate(self.pipeline):
                weight = norm_weights[i]
                layer_norm = layer.perturbation_norm(x=x, lp_style=lp_style)
                if out is None:
                    out = layer_norm * weight
                else:
                    out = out + layer_norm * weight
            return out

    @initialized
    def make_valid_image(self, x):
        x = self.pad(x)
        for layer in self.pipeline:
            layer.make_valid_image(x)
            x = layer(x)


    @initialized
    def constrain_params(self):
        # Need to do some sort of crazy projection operator for general things
        # For now, let's just constrain each thing in sequence

        for layer in self.pipeline:
            layer.constrain_params()

    @initialized
    def update_params(self, step_fxn):
        for layer in self.pipeline:
            layer.update_params(step_fxn)


    @initialized
    def merge_perturbation(self, other, self_mask):
        super(SequentialPerturbation, self).merge_perturbation(other, self_mask)


        new_pipeline = []
        for self_layer, other_layer in zip(self.pipeline, other.pipeline):
            new_pipeline.append(self_layer.merge_perturbation(other_layer,
                                                              self_mask))


        layer_params, global_params = self.perturbation_params

        new_perturbation = SequentialPerturbation(self.threat_model,
                                                layer_params,
                                                global_parameters=global_params,
                                                preinit_pipeline=new_pipeline)
        new_perturbation._merge_setup(self.num_examples)

        return new_perturbation



    def forward(self, x, layer_slice=None):
        """ Layer slice here is either an int or a tuple
        If int, we run forward only the first layer_slice layers
        If tuple, we start at the

        """

        # Blocks to handle only particular layer slices (debugging)
        if layer_slice is None:
            pipeline_iter = iter(self.pipeline)
        elif isinstance(layer_slice, int):
            pipeline_iter = iter(self.pipeline[:layer_slice])
        elif isinstance(layer_slice, tuple):
            pipeline_iter = iter(self.pipeline[layer_slice[0]: layer_slice[1]])
        # End block to handle particular layer slices

        # Handle padding
        original_hw = x.shape[-2:]
        if not self.initialized:
            self.setup(x)

        self.constrain_params()
        self.make_valid_image(x)

        x = self.pad(x)
        for layer in pipeline_iter:
            x = layer(x)
        return self.unpad(x)


    @initialized
    def random_init(self):
        for layer in self.pipeline:
            layer.random_init()

    @initialized
    def attach_originals(self, originals):
        self.originals = originals
        for layer in self.pipeline:
            layer.attach_originals(originals)





