from .matcher import HungarianMatcher
from .criterion import DFINECriterion

from dfine.configs import models
from typing import Literal


def build_criterion(model_variant: Literal['nano', 'small', 'medium', 'large', 'extra_large'],
                    class_mapping: dict[str, dict[str, int]]) -> DFINECriterion:
    criterion_cfg = models[model_variant]
    matcher = HungarianMatcher(**criterion_cfg["matcher"])
    num_classes = max(max(v.values()) if v else 0 for v in class_mapping.values())
    return DFINECriterion(matcher,
                          num_classes=num_classes,
                          **criterion_cfg["DFINECriterion"])
