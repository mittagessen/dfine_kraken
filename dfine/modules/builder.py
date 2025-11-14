from .matcher import HungarianMatcher
from .criterion import DFINECriterion

from dfine.configs import models

def build_criterion(model_size: Literal,
                    class_mapping: dict[str, dict[str, int]]) -> DFINECriterion:
    criterion_cfg = models[model_size]
    matcher = HungarianMatcher(**criterion_cfg["matcher"])
    num_classes = max(max(v.values()) for v in class_mapping.values())
    return DFINECriterion(matcher,
                          num_classes=num_classes,
                          **criterion_cfg["DFINECriterion"])
