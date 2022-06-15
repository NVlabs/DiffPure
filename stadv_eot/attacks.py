# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import functools
import torch
from torch import nn
from torch import optim

# mister_ed
from .recoloradv.mister_ed import loss_functions as lf
from .recoloradv.mister_ed import adversarial_training as advtrain
from .recoloradv.mister_ed import adversarial_perturbations as ap
from .recoloradv.mister_ed import adversarial_attacks as aa
from .recoloradv.mister_ed import spatial_transformers as st


PGD_ITERS = 20


def run_attack_with_random_targets(attack, model, inputs, labels, num_classes):
    """
    Runs an attack with targets randomly selected from all classes besides the
    correct one. The attack should be a function from (inputs, labels) to
    adversarial examples.
    """

    rand_targets = torch.randint(
        0, num_classes - 1, labels.size(),
        dtype=labels.dtype, device=labels.device,
           )
    targets = torch.remainder(labels + rand_targets + 1, num_classes)

    adv_inputs = attack(inputs, targets)
    adv_labels = model(adv_inputs).argmax(1)
    unsuccessful = adv_labels != targets
    adv_inputs[unsuccessful] = inputs[unsuccessful]

    return adv_inputs


class MisterEdAttack(nn.Module):
    """
    Base class for attacks using the mister_ed library.
    """

    def __init__(self, model, threat_model, randomize=False,
                 perturbation_norm_loss=False, lr=0.001, random_targets=False,
                 num_classes=None, **kwargs):
        super().__init__()

        self.model = model
        self.normalizer = nn.Identity()

        self.threat_model = threat_model
        self.randomize = randomize
        self.perturbation_norm_loss = perturbation_norm_loss
        self.attack_kwargs = kwargs
        self.lr = lr
        self.random_targets = random_targets
        self.num_classes = num_classes

        self.attack = None

    def _setup_attack(self):
        cw_loss = lf.CWLossF6(self.model, self.normalizer, kappa=float('inf'))
        if self.random_targets:
            cw_loss.forward = functools.partial(cw_loss.forward, targeted=True)
        perturbation_loss = lf.PerturbationNormLoss(lp=2)
        pert_factor = 0.0
        if self.perturbation_norm_loss is True:
            pert_factor = 0.05
        elif type(self.perturbation_norm_loss) is float:
            pert_factor = self.perturbation_norm_loss
        adv_loss = lf.RegularizedLoss({
            'cw': cw_loss,
            'pert': perturbation_loss,
        }, {
            'cw': 1.0,
            'pert': pert_factor,
        }, negate=True)

        self.pgd_attack = aa.PGD(self.model, self.normalizer,
                                 self.threat_model(), adv_loss)

        attack_params = {
            'optimizer': optim.Adam,
            'optimizer_kwargs': {'lr': self.lr},
            'signed': False,
            'verbose': False,
            'num_iterations': 0 if self.randomize else PGD_ITERS,
            'random_init': self.randomize,
        }
        attack_params.update(self.attack_kwargs)

        self.attack = advtrain.AdversarialAttackParameters(
            self.pgd_attack,
            1.0,
            attack_specific_params={'attack_kwargs': attack_params},
        )
        self.attack.set_gpu(False)

    def forward(self, inputs, labels):
        if self.attack is None:
            self._setup_attack()
        assert self.attack is not None

        if self.random_targets:
            return run_attack_with_random_targets(
                lambda inputs, labels: self.attack.attack(inputs, labels)[0],
                self.model,
                inputs,
                labels,
                num_classes=self.num_classes,
            )
        else:
            return self.attack.attack(inputs, labels)[0]


class StAdvAttack(MisterEdAttack):
    def __init__(self, model, bound=0.05, **kwargs):
        kwargs.setdefault('lr', 0.01)
        super().__init__(
            model,
            threat_model=lambda: ap.ThreatModel(ap.ParameterizedXformAdv, {
                'lp_style': 'inf',
                'lp_bound': bound,
                'xform_class': st.FullSpatial,
                'use_stadv': True,
            }),
            perturbation_norm_loss=0.0025 / bound,
            **kwargs,
        )
