# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from recoloradv.
#
# Source:
# https://github.com/cassidylaidlaw/ReColorAdv/blob/master/recoloradv/utils.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_RECOLORADV).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

from torch import nn
from torch import optim

from .mister_ed.utils.pytorch_utils import DifferentiableNormalize
from .mister_ed import adversarial_perturbations as ap
from .mister_ed import adversarial_attacks as aa
from .mister_ed import spatial_transformers as st
from .mister_ed import loss_functions as lf
from .mister_ed import adversarial_training as advtrain

from . import perturbations as pt
from . import color_transformers as ct
from . import color_spaces as cs


def get_attack_from_name(
    name: str,
    classifier: nn.Module,
    normalizer: DifferentiableNormalize,
    verbose: bool = False,
) -> advtrain.AdversarialAttackParameters:
    """
    Builds an attack from a name like "recoloradv" or "stadv+delta" or
    "recoloradv+stadv+delta".
    """

    threats = []
    norm_weights = []

    for attack_part in name.split('+'):
        if attack_part == 'delta':
            threats.append(ap.ThreatModel(
                ap.DeltaAddition,
                ap.PerturbationParameters(
                    lp_style='inf',
                    lp_bound=8.0 / 255,
                ),
            ))
            norm_weights.append(0.0)
        elif attack_part == 'stadv':
            threats.append(ap.ThreatModel(
                ap.ParameterizedXformAdv,
                ap.PerturbationParameters(
                    lp_style='inf',
                    lp_bound=0.05,
                    xform_class=st.FullSpatial,
                    use_stadv=True,
                ),
            ))
            norm_weights.append(1.0)
        elif attack_part == 'recoloradv':
            threats.append(ap.ThreatModel(
                pt.ReColorAdv,
                ap.PerturbationParameters(
                    lp_style='inf',
                    lp_bound=[0.06, 0.06, 0.06],
                    xform_params={
                        'resolution_x': 16,
                        'resolution_y': 32,
                        'resolution_z': 32,
                    },
                    xform_class=ct.FullSpatial,
                    use_smooth_loss=True,
                    cspace=cs.CIELUVColorSpace(),
                ),
            ))
            norm_weights.append(1.0)
        else:
            raise ValueError(f'Invalid attack "{attack_part}"')

    sequence_threat = ap.ThreatModel(
        ap.SequentialPerturbation,
        threats,
        ap.PerturbationParameters(norm_weights=norm_weights),
    )

    # use PGD attack
    adv_loss = lf.CWLossF6(classifier, normalizer, kappa=float('inf'))
    st_loss = lf.PerturbationNormLoss(lp=2)
    loss_fxn = lf.RegularizedLoss({'adv': adv_loss, 'pert': st_loss},
                                  {'adv': 1.0,      'pert': 0.05},
                                  negate=True)

    pgd_attack = aa.PGD(classifier, normalizer, sequence_threat, loss_fxn)
    return advtrain.AdversarialAttackParameters(
        pgd_attack,
        1.0,
        attack_specific_params={'attack_kwargs': {
            'num_iterations': 100,
            'optimizer': optim.Adam,
            'optimizer_kwargs': {'lr': 0.001},
            'signed': False,
            'verbose': verbose,
        }},
    )
