import torch
import torch.nn as nn

from kraken.models import BaseModel
from kraken.configs import SegmentationInferenceConfig

from os import PathLike
from pathlib import Path
from lightning.fabric import Fabric
from typing import Optional, Literal

from dfine.configs import models
from dfine.modules import HGNetv2, DFINETransformer, HybridEncoder


class DFINE(nn.Module, BaseModel):

    user_metadata = {}
    model_type = 'segmentation'
    _kraken_min_version = '7.0.0'

    def __init__(self, **kwargs):
        """
        A D-FINE object detection model.

        Args:
            model_variant: Literal['nano', 'small', 'medium', 'large', 'extra_large'],
            class_mapping: dict[str, dict[str, int]],
            image_size: tuple[int, int]
        """
        super().__init__()

        if (model_variant := kwargs.get('model_variant', None)) is None:
            raise ValueError('model_variant argument is missing in args.')
        if (class_mapping := kwargs.get('class_mapping', None)) is None:
            raise ValueError('class_mapping argument is missing in args.')
        if (image_size := kwargs.get('image_size', None)) is None:
            raise ValueError('image_size argument is missing in args.')

        self.user_metadata: dict[str, Any] = {'accuracy': [],
                                              'metrics': []}
        self.user_metadata.update(kwargs)

        model_cfg = models[model_variant]
        model_cfg["HybridEncoder"]["eval_spatial_size"] = image_size 
        model_cfg["DFINETransformer"]["eval_spatial_size"] = image_size 
        # rather highest class index 
        num_classes = max(max(v.values()) if v else 0 for v in class_mapping.values()) + 1

        self.backbone = HGNetv2(**model_cfg["HGNetv2"])
        self.encoder = HybridEncoder(**model_cfg["HybridEncoder"])
        self.decoder = DFINETransformer(num_classes=num_classes, **model_cfg["DFINETransformer"])

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

    def prepare_for_inference(self, config: SegmentationInferenceConfig):
        """
        Configures the model for inference.
        """
        self.eval()
        self._inf_config = config

        self._fabric = Fabric(accelerator=self._inf_config.accelerator,
                              devices=self._inf_config.device,
                              precision=self._inf_config.precision)

        self.nn = self._fabric._precision.convert_module(self.nn)
        self.nn = self._fabric.to_device(self.nn)

    @torch.inference_mode()
    def predict(self, im):
        """
        Runs prediction with the model.

        Args:

            > For layout analysis models

            im (PIL.Image.Image): Input image.
        """
        pass
