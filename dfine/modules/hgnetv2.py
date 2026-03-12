"""
HGNetv2 backbone via timm.

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

import timm

from .common import FrozenBatchNorm2d

__all__ = ["HGNetv2"]


class HGNetv2(nn.Module):
    """
    HGNetV2 backbone using timm pretrained models.

    Args:
        name: timm model identifier (e.g. "hgnetv2_b0.ssld_stage1_in22k_in1k")
        return_idx: list of stage indices whose outputs to return
        freeze_stem_only: if True, only freeze the stem (not stages)
        freeze_at: freeze stages up to this index (inclusive); -1 means no stage freezing
        freeze_norm: replace BatchNorm2d with FrozenBatchNorm2d
        pretrained: load timm pretrained weights
    """

    def __init__(self,
                 name,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True):
        super().__init__()
        self.return_idx = return_idx

        model = timm.create_model(name, pretrained=pretrained)
        self.stem = model.stem
        self.stages = model.stages

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
