# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/mister_ed/utils/checkpoints.py
#
# The license for the original version of this file can be
# found in the `recoloradv` directory (LICENSE_RECOLORADV).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

""" Code for saving/loading pytorch models and batches of adversarial images

CHECKPOINT NAMING CONVENTIONS:
    <unique_experiment_name>.<architecture_abbreviation>.<6 digits of epoch number>path.tar
e.g.
    fgsm_def.resnet32.20180301.120000.path.tar

All checkpoints are stored in CHECKPOINT_DIR

Checkpoints are state dicts only!!!

"""

import torch
import os
import re
import glob
from .. import config
import numpy as np
import random

CHECKPOINT_DIR = config.MODEL_PATH
OUTPUT_IMAGE_DIR = config.OUTPUT_IMAGE_PATH


##############################################################################
#                                                                            #
#                               CHECKPOINTING MODELS                         #
#                                                                            #
##############################################################################


def clear_experiment(experiment_name, architecture):
    """ Deletes all saved state dicts for an experiment/architecture pair """

    for filename in params_to_filename(experiment_name, architecture):
        full_path = os.path.join(*[CHECKPOINT_DIR, filename])
        os.remove(full_path) if os.path.exists(full_path) else None


def list_saved_epochs(experiment_name, architecture):
    """ Returns a list of int epochs we've checkpointed for this
        experiment name and architecture
    """

    extract_epoch = lambda f: int(f.split('.')[-3])
    filename_list = params_to_filename(experiment_name, architecture)
    return [extract_epoch(f) for f in filename_list]


def params_to_filename(experiment_name, architecture, epoch_val=None):
    """ Outputs string name of file.
    ARGS:
        experiment_name : string - name of experiment we're saving
        architecture : string - abbreviation for model architecture
        epoch_val : int/(intLo, intHi)/None -
                    - if int we return this int exactly
                    - if (intLo, intHi) we return all existing filenames with
                      highest epoch in range (intLo, intHi), in sorted order
                    - if None, we return all existing filenames with params
                      in ascending epoch-sorted order

    RETURNS:
        filenames: string or (possibly empty) string[] of just the base name
        of saved models
    """

    if isinstance(epoch_val, int):
        return '.'.join([experiment_name, architecture, '%06d' % epoch_val,
                         'path', 'tar'])

    glob_prefix = os.path.join(*[CHECKPOINT_DIR,
                                 '%s.%s.*' % (experiment_name, architecture)])
    re_prefix = '%s\.%s\.' % (experiment_name, architecture)
    re_suffix = r'\.path\.tar'

    valid_name = lambda f: bool(re.match(re_prefix + r'\d{6}' + re_suffix, f))
    select_epoch = lambda f: int(re.sub(re_prefix, '',
                                        re.sub(re_suffix, '', f)))
    valid_epoch = lambda e: (e >= (epoch_val or (0, 0))[0] and
                             e <= (epoch_val or (0, float('inf')))[1])

    filename_epoch_pairs = []
    for full_path in glob.glob(glob_prefix):
        filename = os.path.basename(full_path)
        if not valid_name(filename):
            continue

        epoch = select_epoch(filename)
        if valid_epoch(epoch):
            filename_epoch_pairs.append((filename, epoch))

    return [_[0] for _ in sorted(filename_epoch_pairs, key=lambda el: el[1])]


def save_state_dict(experiment_name, architecture, epoch_val, model,
                    k_highest=10):
    """ Saves the state dict of a model with the given parameters.
    ARGS:
        experiment_name : string - name of experiment we're saving
        architecture : string - abbreviation for model architecture
        epoch_val : int - which epoch we're saving
        model : model - object we're saving the state dict of
        k_higest : int - if not None, we make sure to not include more than
                         k state_dicts for (experiment_name, architecture) pair,
                         keeping the k-most recent if we overflow
    RETURNS:
        The model we saved
    """

    # First resolve THIS filename
    this_filename = params_to_filename(experiment_name, architecture, epoch_val)

    # Next clear up memory if too many state dicts
    current_filenames = params_to_filename(experiment_name, architecture)
    delete_els = []
    if k_highest is not None:
        num_to_delete = len(current_filenames) - k_highest + 1
        if num_to_delete > 0:
            delete_els = sorted(current_filenames)[:num_to_delete]

    for delete_el in delete_els:
        full_path = os.path.join(*[CHECKPOINT_DIR, delete_el])
        os.remove(full_path) if os.path.exists(full_path) else None

    # Finally save the state dict
    torch.save(model.state_dict(), os.path.join(*[CHECKPOINT_DIR,
                                                  this_filename]))

    return model


def load_state_dict_from_filename(filename, model):
    """ Skips the whole parameter argument thing and just loads the whole
        state dict from a filename.
    ARGS:
        filename : string - filename without directories
        model : nn.Module - has 'load_state_dict' method
    RETURNS:
        the model loaded with the weights contained in the file
    """
    assert len(glob.glob(os.path.join(*[CHECKPOINT_DIR, filename]))) == 1

    # LOAD FILENAME

    # If state_dict in keys, use that as the loader
    right_dict = lambda d: d.get('state_dict', d)

    model.load_state_dict(right_dict(torch.load(
        os.path.join(*[CHECKPOINT_DIR, filename]))))
    return model


def load_state_dict(experiment_name, architecture, epoch, model):
    """ Loads a checkpoint that was previously saved
        experiment_name : string - name of experiment we're saving
        architecture : string - abbreviation for model architecture
        epoch_val : int - which epoch we're loading
    """

    filename = params_to_filename(experiment_name, architecture, epoch)
    return load_state_dict_from_filename(filename, model)


###############################################################################
#                                                                             #
#                              CHECKPOINTING DATA                             #
#                                                                             #
###############################################################################
"""
    This is a hacky fix to save batches of adversarial images along with their
    labels.
"""


class CustomDataSaver(object):
    # TODO: make this more pytorch compliant
    def __init__(self, image_subdirectory):
        self.image_subdirectory = image_subdirectory
        # make this folder if it doesn't exist yet

    def save_minibatch(self, examples, labels):
        """ Assigns a random name to this minibatch and saves the examples and
            labels in two separate files:
            <random_name>.examples.npy and <random_name>.labels.npy
        ARGS:
            examples: Variable or Tensor (NxCxHxW) - examples to be saved
            labels : Variable or Tensor (N) - labels matching the examples
        """
        # First make both examples and labels into numpy arrays
        examples = examples.cpu().numpy()
        labels = labels.cpu().numpy()

        # Make a name for the files
        random_string = str(random.random())[2:]  # DO THIS BETTER WHEN I HAVE INTERNET

        # Save both files
        example_file = '%s.examples.npy' % random_string
        example_path = os.path.join(OUTPUT_IMAGE_DIR, self.image_subdirectory,
                                    example_file)
        np.save(example_path, examples)

        label_file = '%s.labels.npy' % random_string
        label_path = os.path.join(OUTPUT_IMAGE_DIR, self.image_subdirectory,
                                  label_file)
        np.save(label_path, labels)


class CustomDataLoader(object):
    # TODO: make this more pytorch compliant
    def __init__(self, image_subdirectory, batch_size=128, to_tensor=True,
                 use_gpu=False):
        super(CustomDataLoader, self).__init__()
        self.image_subdirectory = image_subdirectory
        self.batch_size = batch_size

        assert to_tensor >= use_gpu
        self.to_tensor = to_tensor
        self.use_gpu = use_gpu

    def _prepare_data(self, examples, labels):
        """ Takes in numpy examples and labels and tensor-ifies and cuda's them
            if necessary
        """

        if self.to_tensor:
            examples = torch.Tensor(examples)
            labels = torch.Tensor(labels)

        if self.use_gpu:
            examples = examples.cuda()
            labels = labels.cuda()

        return (examples, labels)

    def _base_loader(self, prefix, which):
        assert which in ['examples', 'labels']
        filename = '%s.%s.npy' % (prefix, which)
        full_path = os.path.join(OUTPUT_IMAGE_DIR, self.image_subdirectory,
                                 filename)
        return np.load(full_path)

    def _example_loader(self, prefix):
        """ Loads the numpy array of examples given the random 'prefix' """
        return self._base_loader(prefix, 'examples')

    def _label_loader(self, prefix):
        """ Loads the numpy array of labels given the random 'prefix' """
        return self._base_loader(prefix, 'labels')

    def __iter__(self):

        # First collect all the filenames:
        glob_prefix = os.path.join(OUTPUT_IMAGE_DIR, self.image_subdirectory,
                                   '*')
        files = glob.glob(glob_prefix)
        valid_random_names = set(os.path.basename(_).split('.')[0]
                                 for _ in files)

        # Now loop through filenames and yield out minibatches of correct size
        running_examples, running_labels = [], []
        running_size = 0
        for random_name in valid_random_names:
            # Load data from files and append to 'running' lists
            loaded_examples = self._example_loader(random_name)
            loaded_labels = self._label_loader(random_name)
            running_examples.append(loaded_examples)
            running_labels.append(loaded_labels)
            running_size += loaded_examples.shape[0]

            if running_size < self.batch_size:
                # Load enough data to populate one minibatch, which might
                # take multiple files
                continue

            # Concatenate all images together
            merged_examples = np.concatenate(running_examples, axis=0)
            merged_labels = np.concatenate(running_labels, axis=0)

            # Make minibatches out of concatenated things,
            for batch_no in range(running_size // self.batch_size):
                index_lo = batch_no * self.batch_size
                index_hi = index_lo + self.batch_size
                example_batch = merged_examples[index_lo:index_hi]
                label_batch = merged_labels[index_lo:index_hi]
                yield self._prepare_data(example_batch, label_batch)

            # Handle any remainder for remaining files
            remainder_idx = (running_size // self.batch_size) * self.batch_size
            running_examples = [merged_examples[remainder_idx:]]
            running_labels = [merged_labels[remainder_idx:]]
            running_size = running_size - remainder_idx

        # If we're out of files, yield this last sub-minibatch of data
        if running_size > 0:
            merged_examples = np.concatenate(running_examples, axis=0)
            merged_labels = np.concatenate(running_labels, axis=0)
            yield self._prepare_data(merged_examples, merged_labels)
