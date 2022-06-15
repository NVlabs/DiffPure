# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/mister_ed/utils/discretization.py
#
# The license for the original version of this file can be
# found in the `recoloradv` directory (LICENSE_RECOLORADV).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

""" File that holds techniques for discretizing images --
    In general, images of the form NxCxHxW will with values in the [0.,1.] range
    need to be converted to the [0, 255 (int)] range to be displayed as images.

    Sometimes the naive rounding scheme can mess up the classification, so this
    file holds techniques to discretize these images into tensors with values
    of the form i/255.0 for some integers i.
"""

import torch
from torch.autograd import Variable
from . import pytorch_utils as utils


##############################################################################
#                                                                            #
#                       HELPER METHODS                                       #
#                                                                            #
##############################################################################


def discretize_image(img_tensor, zero_one=False):
    """ Discretizes an image tensor into a tensor filled with ints ranging
        between 0 and 255
    ARGS:
        img_tensor : floatTensor (NxCxHxW) - tensor to be discretized
        pixel_max : int - discretization bucket size
        zero_one : bool - if True divides output by 255 before returning it
    """

    assert float(torch.min(img_tensor)) >= 0.
    assert float(torch.max(img_tensor)) <= 1.0

    original_shape = img_tensor.shape
    if img_tensor.dim() != 4:
        img_tensor = img_tensor.unsqueeze(0)

    int_tensors = []  # actually floatTensor, but full of ints
    img_shape = original_shape[1:]
    for example in img_tensor:
        pixel_channel_tuples = zip(*list(smp.toimage(example).getdata()))
        int_tensors.append(img_tensor.new(pixel_channel_tuples).view(img_shape))

    stacked_tensors = torch.stack(int_tensors)
    if zero_one:
        return stacked_tensors / 255.0
    return stacked_tensors


##############################################################################
#                                                                            #
#                        MAIN DISCRETIZATION TECHNIQUES                      #
#                                                                            #
##############################################################################

def discretized_adversarial(img_tensor, classifier_net, normalizer,
                            flavor='greedy'):
    """ Takes in an image_tensor and classifier/normalizer pair and outputs a
        'discretized' image_tensor [each val is i/255.0 for some integer i]
        with the same classification
    ARGS:
        img_tensor : tensor (NxCxHxW) - tensor of images with values between
                     0.0 and 1.0.
        classifier_net : NN - neural net with .forward method to classify
                         normalized images
        normalizer : differentiableNormalizer object - normalizes 0,1 images
                     into classifier_domain
        flavor : string - either 'random' or 'greedy', determining which
                 'next_pixel_to_flip' function we use
    RETURNS:
        img_tensor of the same shape, but no with values of the form i/255.0
        for integers i.
    """

    img_tensor = utils.safe_tensor(img_tensor)

    nptf_map = {'random': flip_random_pixel,
                'greedy': flip_greedy_pixel}
    next_pixel_to_flip = nptf_map[flavor](classifier_net, normalizer)

    ##########################################################################
    # First figure out 'correct' labels and the 'discretized' labels         #
    ##########################################################################
    var_img = utils.safe_var(img_tensor)
    norm_var = normalizer.forward(var_img)
    norm_output = classifier_net.forward(norm_var)
    correct_targets = norm_output.max(1)[1]

    og_discretized = utils.safe_var(discretize_image(img_tensor, zero_one=True))
    norm_discretized = normalizer.forward(og_discretized)
    discretized_output = classifier_net.forward(norm_discretized)
    discretized_targets = discretized_output.max(1)[1]

    ##########################################################################
    # Collect idxs for examples affected by discretization                   #
    ##########################################################################
    incorrect_idxs = set()

    for i, el in enumerate(correct_targets.ne(discretized_targets)):
        if float(el) != 0:
            incorrect_idxs.add(i)

    ##########################################################################
    #   Fix all bad images                                                   #
    ##########################################################################

    corrected_imgs = []
    for idx in incorrect_idxs:
        desired_target = correct_targets[idx]
        example = og_discretized[idx].data.clone()  # tensor
        signs = torch.sign(var_img - og_discretized)
        bad_discretization = True
        pixels_changed_so_far = set()  # populated with tuples of idxs

        while bad_discretization:
            pixel_idx, grad_sign = next_pixel_to_flip(example,
                                                      pixels_changed_so_far,
                                                      desired_target)
            pixels_changed_so_far.add(pixel_idx)

            if grad_sign == 0:
                grad_sign = utils.tuple_getter(signs[idx], pixel_idx)

            new_val = (grad_sign / 255. + utils.tuple_getter(example, pixel_idx))
            utils.tuple_setter(example, pixel_idx, float(new_val))

            new_out = classifier_net.forward(normalizer.forward( \
                Variable(example.unsqueeze(0))))
            bad_discretization = (int(desired_target) != int(new_out.max(1)[1]))
        corrected_imgs.append(example)

    # Stack up results
    output = []

    for idx in range(len(img_tensor)):
        if idx in incorrect_idxs:
            output.append(corrected_imgs.pop(0))
        else:
            output.append(og_discretized[idx].data)

    return torch.stack(output)  # Variable


#############################################################################
#                                                                           #
#                       FLIP TECHNIQUES                                     #
#                                                                           #
#############################################################################
''' Flip techniques in general have the following specs:
    ARGS:
        classifier_net : NN - neural net with .forward method to classify
                         normalized images
        normalizer : differentiableNormalizer object - normalizes 0,1 images
                     into classifier_domain
    RETURNS: flip_function
'''

'''
    Flip function is a function that takes the following args:
    ARGS:
        img_tensor : Tensor (CxHxW) - image tensor in range 0.0 to 1.0 and is
                     already discretized
        pixels_changed_so_far: set - set of index_tuples that have already been
                               modified (we don't want to modify a pixel by
                               more than 1/255 in any channel)
        correct_target : torch.LongTensor (1) - single element in a tensor that
                         is the target class
                         (e.g. int between 0 and 9 for CIFAR )
    RETURNS: (idx_tuple, sign)
        index_tuple is a triple of indices indicating which pixel-channel needs
        to be modified, and sign is in {-1, 0, 1}. If +-1, we will modify the
        pixel-channel in that direction, otherwise we'll modify in the opposite
        of the direction that discretization rounded to.
'''


def flip_random_pixel(classifier_net, normalizer):
    def flip_fxn(img_tensor, pixels_changed_so_far, correct_target):
        numel = img_tensor.numel()
        if len(pixels_changed_so_far) > numel * .9:
            raise Exception("WHAT IS GOING ON???")

        while True:
            pixel_idx, _ = utils.random_element_index(img_tensor)
            if pixel_idx not in pixels_changed_so_far:
                return pixel_idx, 0

    return flip_fxn


def flip_greedy_pixel(classifier_net, normalizer):
    def flip_fxn(img_tensor, pixels_changed_so_far, correct_target,
                 classifier_net=classifier_net, normalizer=normalizer):
        # Computes gradient and figures out which px most affects class_out
        classifier_net.zero_grad()
        img_var = Variable(img_tensor.unsqueeze(0), requires_grad=True)
        class_out = classifier_net.forward(normalizer.forward(img_var))

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(class_out, correct_target)  # RESHAPE HERE
        loss.backward()
        # Really inefficient algorithm here, can probably do better
        new_grad_data = img_var.grad.data.clone().squeeze()
        signs = new_grad_data.sign()
        for idx_tuple in pixels_changed_so_far:
            utils.tuple_setter(new_grad_data, idx_tuple, 0)

        argmax = utils.torch_argmax(new_grad_data.abs())
        return argmax, -1 * utils.tuple_getter(signs, argmax)

    return flip_fxn
