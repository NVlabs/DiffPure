# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/mister_ed/adversarial_training.py
#
# The license for the original version of this file can be
# found in the `recoloradv` directory (LICENSE_RECOLORADV).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

""" Contains training code for adversarial training """

from __future__ import print_function
import torch
import torch.cuda as cuda
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable

import random

from .utils import pytorch_utils as utils, checkpoints


##############################################################################
#                                                                            #
#                               ATTACK PARAMETERS OBJECT                     #
#                                                                            #
##############################################################################

class AdversarialAttackParameters(object):
    """ Wrapper to store an adversarial attack object as well as some extra
        parameters for how to use it in training
    """

    def __init__(self, adv_attack_obj, proportion_attacked=1.0,
                 attack_specific_params=None):
        """ Stores params for how to use adversarial attacks in training
        ARGS:
            adv_attack_obj : AdversarialAttack subclass -
                             thing that actually does the attack
            proportion_attacked: float between [0.0, 1.0] - what proportion of
                                 the minibatch we build adv examples for
            attack_specific_params: possibly None dict, but possibly dict with
                                    specific parameters for attacks

        """
        self.adv_attack_obj = adv_attack_obj
        self.proportion_attacked = proportion_attacked

        attack_specific_params = attack_specific_params or {}
        self.attack_specific_params = attack_specific_params
        self.attack_kwargs = attack_specific_params.get('attack_kwargs', {})

    def set_gpu(self, use_gpu):
        """ Propagates changes of the 'use_gpu' parameter down to the attack
        ARGS:
            use_gpu : bool - if True, the attack uses the GPU, ow it doesn't
        RETURNS:
            None
        """
        self.adv_attack_obj.use_gpu = use_gpu

    def attack(self, inputs, labels):
        """ Builds some adversarial examples given real inputs and labels
        ARGS:
            inputs : torch.Tensor (NxCxHxW) - tensor with examples needed
            labels : torch.Tensor (N) - tensor with the examples needed
        RETURNS:
            some sample of (self.proportion_attacked * N ) examples that are
            adversarial, and the corresponding NONADVERSARIAL LABELS

            output is a tuple with three tensors:
             (adv_examples, pre_adv_labels, selected_idxs, coupled )
             adv_examples: Tensor with shape (N'xCxHxW) [the perturbed outputs]
             pre_adv_labels: Tensor with shape (N') [original labels]
             selected_idxs : Tensor with shape (N') [idxs selected]
             adv_inputs : Tensor with shape (N') [examples used to make advs]
             perturbation: Adversarial Perturbation Object
        """
        num_elements = inputs.shape[0]

        selected_idxs = sorted(random.sample(list(range(num_elements)),
                                             int(self.proportion_attacked * num_elements)))

        selected_idxs = inputs.new(selected_idxs).long()
        if selected_idxs.numel() == 0:
            return (None, None, None)

        adv_inputs = Variable(inputs.index_select(0, selected_idxs))
        pre_adv_labels = labels.index_select(0, selected_idxs)

        perturbation = self.adv_attack_obj.attack(adv_inputs.data,
                                                  pre_adv_labels,
                                                  **self.attack_kwargs)
        adv_examples = perturbation(adv_inputs)

        return (adv_examples, pre_adv_labels, selected_idxs, adv_inputs,
                perturbation)

    def eval(self, ground_inputs, adv_inputs, labels, idxs, topk=1):
        """ Outputs the accuracy of the adversarial examples

            NOTE: notice the difference between N and N' in the argument
        ARGS:
            ground_inputs: Variable (NxCxHxW) - examples before we did
                           adversarial perturbation. Vals in [0, 1] range
            adversarials: Variable (N'xCxHxW) - examples after we did
                           adversarial perturbation. Should be same shape and
                           in same order as ground_truth
            labels: Variable (longTensor N) - correct labels of classification
                    output
            idxs: Variable (longtensor N') - indices of ground_inputs/labels
                  used for adversarials.
        RETURNS:
            tuple of (% of correctly classified original examples,
                      % of correctly classified adversarial examples)

        """

        selected_grounds = ground_inputs.index_select(0, idxs)
        selected_labels = labels.index_select(0, idxs)
        return self.adv_attack_obj.eval(selected_grounds, adv_inputs,
                                        selected_labels, topk=topk)

    def eval_attack_only(self, adv_inputs, labels, topk=1):
        """ Outputs the accuracy of the adv_inputs only
        ARGS:
            adv_inputs: Variable NxCxHxW - examples after we did adversarial
                                           perturbation
            labels: Variable (longtensor N) - correct labels of classification
                                              output
            topk: int - criterion for 'correct' classification
        RETURNS:
            (int) number of correctly classified examples
        """

        return self.adv_attack_obj.eval_attack_only(adv_inputs, labels,
                                                    topk=topk)


##############################################################################
#                                                                            #
#                               TRAINING OBJECT                              #
#                                                                            #
##############################################################################


class AdversarialTraining(object):
    """ Wrapper for training of a NN with adversarial examples cooked in
    """

    def __init__(self, classifier_net, normalizer,
                 experiment_name, architecture_name,
                 manual_gpu=None):

        """
        ARGS:
        classifier_net : nn.Module subclass - instance of neural net to classify
                         images. Can have already be trained, or not
        normalizer : DifferentiableNormalize - object to convert to zero-mean
                     unit-variance domain
        experiment_name : String - human-readable name of the 'trained_model'
                          (this is helpful for identifying checkpoints later)
        manual_gpu : None or bool - if not None is a manual override of whether
                     or not to use the GPU. If left None, we use the GPU if we
                     can

        ON NOMENCLATURE:
        Depending on verbosity levels, training checkpoints are saved after
        some training epochs. These are saved as
        '<experiment_name>/<architecture_name>/<epoch>.path.tar'

        Best practice is to keep architecture_name consistent across
        adversarially trained models built off the same architecture and having
        a descriptive experiment_name for each training instance
        """
        self.classifier_net = classifier_net
        self.normalizer = normalizer
        self.experiment_name = experiment_name
        self.architecture_name = architecture_name

        if manual_gpu is not None:
            self.use_gpu = manual_gpu
        else:
            self.use_gpu = utils.use_gpu()

        self.verbosity_level = None
        self.verbosity_minibatch = None
        self.verbosity_adv = None
        self.verbosity_epoch = None

        self.logger = utils.TrainingLogger()
        self.log_level = None
        self.log_minibatch = None
        self.log_adv = None
        self.log_epoch = None

    def reset_logger(self):
        """ Clears the self.logger instance - useful occasionally """
        self.logger = utils.TrainingLogger()

    def set_verbosity_loglevel(self, level,
                               verbosity_or_loglevel='verbosity'):
        """ Sets the verbosity or loglevel for training.
            Is called in .train method so this method doesn't need to be
            explicitly called.

            Verbosity is mapped from a string to a comparable int 'level'.
            <val>_level : int - comparable value of verbosity
            <val>_minibatch: int - we do a printout every this many
                                       minibatches
            <val>_adv: int - we evaluate the efficacy of our attack every
                                 this many minibatches
            <val>_epoch: int - we printout/log and checkpoint every this many
                               epochs
        ARGS:
            level : string ['low', 'medium', 'high', 'snoop'],
                        varying levels of verbosity/logging in increasing order

        RETURNS: None
        """
        assert level in ['low', 'medium', 'high', 'snoop']
        assert verbosity_or_loglevel in ['verbosity', 'loglevel']
        setattr(self, verbosity_or_loglevel, level)

        _level = {'low': 0,
                  'medium': 1,
                  'high': 2,
                  'snoop': 420}[level]
        setattr(self, verbosity_or_loglevel + '_level', _level)

        _minibatch = {'medium': 2000,
                      'high': 100,
                      'snoop': 1}.get(level)
        setattr(self, verbosity_or_loglevel + '_minibatch', _minibatch)

        _adv = {'medium': 2000,
                'high': 100,
                'snoop': 1}.get(level)
        setattr(self, verbosity_or_loglevel + '_adv', _minibatch)

        _epoch = {'low': 100,
                  'medium': 10,
                  'high': 1,
                  'snoop': 1}.get(level)
        setattr(self, verbosity_or_loglevel + '_epoch', _epoch)

    def _attack_subroutine(self, attack_parameters, inputs, labels,
                           epoch_num, minibatch_num, adv_saver,
                           logger):
        """ Subroutine to run the specified attack on a minibatch and append
            the results to inputs/labels.

        NOTE: THIS DOES NOT MUTATE inputs/labels !!!!

        ARGS:
            attack_parameters:  {k: AdversarialAttackParameters} (or None) -
                                if not None, contains info on how to do adv
                                attacks. If None, we don't train adversarially
            inputs : Tensor (NxCxHxW) - minibatch of data we build adversarial
                                        examples for
            labels : Tensor (longtensor N) - minibatch of labels
            epoch_num : int - number of which epoch we're working on.
                        Is helpful for printing
            minibatch_num : int - number of which minibatch we're working on.
                            Is helpful for printing
            adv_saver : None or checkpoints.CustomDataSaver -
                        if not None, we save the adversarial images for later
                        use, else we don't save them.
            logger : utils.TrainingLogger instance -  logger instance to keep
                     track of logging data if we need data for this instance
        RETURNS:
            (inputs, labels, adv_inputs, coupled_inputs)
            where inputs = <arg inputs> ++ adv_inputs
                  labels is original labels
                  adv_inputs is the (Variable) adversarial examples generated,
                  coupled_inputs is the (Variable) inputs used to generate the
                                 adversarial examples (useful for when we don't
                                 augment 1:1).
        """
        if attack_parameters is None:
            return inputs, labels, None, None

        assert isinstance(attack_parameters, dict)

        adv_inputs_total, adv_labels_total, coupled_inputs = [], [], []
        for (key, param) in attack_parameters.items():
            adv_data = param.attack(inputs, labels)
            adv_inputs, adv_labels, adv_idxs, og_adv_inputs, _ = adv_data

            needs_print = (self.verbosity_level >= 1 and
                           minibatch_num % self.verbosity_adv == self.verbosity_adv - 1)
            needs_log = (self.loglevel_level >= 1 and
                         minibatch_num % self.loglevel_adv == self.loglevel_adv - 1)

            if needs_print or needs_log:
                accuracy = param.eval(inputs, adv_inputs, labels, adv_idxs)

            if needs_print:
                print('[%d, %5d] accuracy: (%.3f, %.3f)' %
                      (epoch_num, minibatch_num + 1, accuracy[1], accuracy[0]))

            if needs_log:
                logger.log(key, epoch_num, minibatch_num + 1,
                           (accuracy[1], accuracy[0]))

            if adv_saver is not None:  # Save the adversarial examples
                adv_saver.save_minibatch(adv_inputs, adv_labels)

            adv_inputs_total.append(adv_inputs)
            adv_labels_total.append(adv_labels)
            coupled_inputs.append(og_adv_inputs)

        inputs = torch.cat([inputs] + [_.data for _ in adv_inputs_total], dim=0)
        labels = torch.cat([labels] + adv_labels_total, dim=0)
        coupled = torch.cat(coupled_inputs, dim=0)
        return inputs, labels, torch.cat(adv_inputs_total, dim=0), coupled

    def train(self, data_loader, num_epochs, train_loss,
              optimizer=None, attack_parameters=None,
              verbosity='medium', loglevel='medium', logger=None,
              starting_epoch=0, adversarial_save_dir=None,
              regularize_adv_scale=None):
        """ Modifies the NN weights of self.classifier_net by training with
            the specified parameters s
        ARGS:
            data_loader: torch.utils.data.DataLoader OR
                         checkpoints.CustomDataLoader - object that loads the
                         data
            num_epoch: int - number of epochs to train on
            train_loss: ????  - TBD
            optimizer: torch.Optimizer subclass - defaults to Adam with some
                       decent default params. Pass this in as an actual argument
                       to do anything different
            attack_parameters:  AdversarialAttackParameters obj | None |
                                {key: AdversarialAttackParameters} -
                                if not None, is either an object or dict of
                                objects containing names and info on how to do
                                adv attacks. If None, we don't train
                                adversarially
            verbosity : string - must be 'low', 'medium', 'high', which
                        describes how much to print
            loglevel : string - must be 'low', 'medium', 'high', which
                        describes how much to log
            logger : if not None, is a utils.TrainingLogger instance. Otherwise
                     we use this instance's self.logger object to log
            starting_epoch : int - which epoch number we start on. Is useful
                             for correct labeling of checkpoints and figuring
                             out how many epochs we actually need to run for
                             (i.e., num_epochs - starting_epoch)
            adversarial_save_dir: string or None - if not None is the name of
                                  the directory we save adversarial images to.
                                  If None, we don't save adversarial images
            regularize_adv_scale : float > 0 or None - if not None we do L1 loss
                                   between the logits of the adv examples and
                                   the inputs used to generate them. This is the
                                   scale constant of that loss
            stdout_prints: bool - if True we print out using stdout so we don't
                                  spam logs like crazy

        RETURNS:
            None, but modifies the classifier_net's weights
        """

        ######################################################################
        #   Setup/ input validations                                         #
        ######################################################################
        self.classifier_net.train()  # in training mode
        assert isinstance(num_epochs, int)

        if attack_parameters is not None:
            if not isinstance(attack_parameters, dict):
                attack_parameters = {'attack': attack_parameters}

            # assert that the adv attacker uses the NN that's being trained
            for param in attack_parameters.values():
                assert (param.adv_attack_obj.classifier_net ==
                        self.classifier_net)

        assert not (self.use_gpu and not cuda.is_available())
        if self.use_gpu:
            self.classifier_net.cuda()
        if attack_parameters is not None:
            for param in attack_parameters.values():
                param.set_gpu(self.use_gpu)

        # Verbosity parameters
        assert verbosity in ['low', 'medium', 'high', 'snoop', None]
        self.set_verbosity_loglevel(verbosity,
                                    verbosity_or_loglevel='verbosity')
        verbosity_level = self.verbosity_level
        verbosity_minibatch = self.verbosity_minibatch
        verbosity_epoch = self.verbosity_epoch

        # Loglevel parameters and logger initialization
        assert loglevel in ['low', 'medium', 'high', 'snoop', None]
        if logger is None:
            logger = self.logger
        if logger.data_count() > 0:
            print("WARNING: LOGGER IS NOT EMPTY! BE CAREFUL!")
        logger.add_series('training_loss')
        for key in (attack_parameters or {}).keys():
            logger.add_series(key)

        self.set_verbosity_loglevel(loglevel, verbosity_or_loglevel='loglevel')
        loglevel_level = self.loglevel_level
        loglevel_minibatch = self.loglevel_minibatch
        loglevel_epoch = self.loglevel_epoch

        # Adversarial image saver:
        adv_saver = None
        if adversarial_save_dir is not None and attack_parameters is not None:
            adv_saver = checkpoints.CustomDataSaver(adversarial_save_dir)

        # setup loss fxn, optimizer
        optimizer = optimizer or optim.Adam(self.classifier_net.parameters(),
                                            lr=0.001)

        # setup regularize adv object
        if regularize_adv_scale is not None:
            regularize_adv_criterion = nn.L1Loss()

        ######################################################################
        #   Training loop                                                    #
        ######################################################################

        for epoch in range(starting_epoch + 1, num_epochs + 1):
            running_loss_print, running_loss_print_mb = 0.0, 0
            running_loss_log, running_loss_log_mb = 0.0, 0
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data
                if self.use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # Build adversarial examples
                attack_out = self._attack_subroutine(attack_parameters,
                                                     inputs, labels,
                                                     epoch, i, adv_saver,
                                                     logger)
                inputs, labels, adv_examples, adv_inputs = attack_out
                # Now proceed with standard training
                self.normalizer.differentiable_call()
                self.classifier_net.train()
                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()

                # forward step
                outputs = self.classifier_net.forward(self.normalizer(inputs))
                loss = train_loss.forward(outputs, labels)

                if regularize_adv_scale is not None:
                    # BE SURE TO 'DETACH' THE ADV_INPUTS!!!
                    reg_adv_loss = regularize_adv_criterion(adv_examples,
                                                            Variable(adv_inputs.data))
                    print(float(loss), regularize_adv_scale * float(reg_adv_loss))
                    loss = loss + regularize_adv_scale * reg_adv_loss

                # backward step
                loss.backward()
                optimizer.step()

                # print things

                running_loss_print += float(loss.data)
                running_loss_print_mb += 1
                if (verbosity_level >= 1 and
                        i % verbosity_minibatch == verbosity_minibatch - 1):
                    print('[%d, %5d] loss: %.6f' %
                          (epoch, i + 1, running_loss_print /
                           float(running_loss_print_mb)))
                    running_loss_print = 0.0
                    running_loss_print_mb = 0

                # log things
                running_loss_log += float(loss.data)
                running_loss_log_mb += 1
                if (loglevel_level >= 1 and
                        i % loglevel_minibatch == loglevel_minibatch - 1):
                    logger.log('training_loss', epoch, i + 1,
                               running_loss_log / float(running_loss_log_mb))
                    running_loss_log = 0.0
                    running_loss_log_mb = 0

            # end_of_epoch
            if epoch % verbosity_epoch == 0:
                print("COMPLETED EPOCH %04d... checkpointing here" % epoch)
                checkpoints.save_state_dict(self.experiment_name,
                                            self.architecture_name,
                                            epoch, self.classifier_net,
                                            k_highest=3)

        if verbosity_level >= 1:
            print('Finished Training')

        return logger

    def train_from_checkpoint(self, data_loader, num_epochs, loss_fxn,
                              optimizer=None, attack_parameters=None,
                              verbosity='medium',
                              starting_epoch='max',
                              adversarial_save_dir=None):
        """ Resumes training from a saved checkpoint with the same architecture.
            i.e. loads weights from specified checkpoint, figures out which
                 epoch we checkpointed on and then continues training until
                 we reach num_epochs epochs
        ARGS:
            same as in train
            starting_epoch: 'max' or int - which epoch we start training from.
                             'max' means the highest epoch we can find,
                             an int means a specified int epoch exactly.
        RETURNS:
            None
        """

        ######################################################################
        #   Checkpoint handling block                                        #
        ######################################################################
        # which epoch to load
        valid_epochs = checkpoints.list_saved_epochs(self.experiment_name,
                                                     self.architecture_name)
        assert valid_epochs != []
        if starting_epoch == 'max':
            epoch = max(valid_epochs)
        else:
            assert starting_epoch in valid_epochs
            epoch = starting_epoch

        # modify the classifer to use these weights

        self.classifier_net = checkpoints.load_state_dict(self.experiment_name,
                                                          self.architecture_name,
                                                          epoch,
                                                          self.classifier_net)

        ######################################################################
        #   Training block                                                   #
        ######################################################################

        self.train(data_loader, num_epochs, loss_fxn,
                   optimizer=optimizer,
                   attack_parameters=attack_parameters,
                   verbosity=verbosity,
                   starting_epoch=epoch,
                   adversarial_save_dir=adversarial_save_dir)
