import uuid
import torch
import torch.nn as nn
import torch.nn.init as init

from lightning.fabric import Fabric
from typing import Any, TYPE_CHECKING, Optional
import copy

from kraken.models import SegmentationBaseModel

from dfine.configs import models
from dfine.modules import HGNetv2, DFINETransformer, HybridEncoder

if TYPE_CHECKING:
    from PIL import Image
    from kraken.containers import Segmentation
    from kraken.configs import SegmentationInferenceConfig


def _normalize_class_mapping(class_mapping: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {'lines': dict(class_mapping.get('lines', {})),
            'regions': dict(class_mapping.get('regions', {}))}


class DFINEModel(nn.Module, SegmentationBaseModel):

    model_type = ['segmentation']
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
        if kwargs.get('num_top_queries', None) is None:
            raise ValueError('num_top_queries argument is missing in args.')

        class_mapping = _normalize_class_mapping(class_mapping)
        kwargs['class_mapping'] = class_mapping

        self.user_metadata.update({'accuracy': [], 'metrics': []})
        self.user_metadata.update(kwargs)

        model_cfg = copy.deepcopy(models[model_variant])
        model_cfg["HybridEncoder"]["eval_spatial_size"] = image_size
        model_cfg["DFINETransformer"]["eval_spatial_size"] = image_size

        self.num_classes = max(max(v.values()) if v else 0 for v in class_mapping.values()) + 1

        self.backbone = HGNetv2(**model_cfg["HGNetv2"])
        self.encoder = HybridEncoder(**model_cfg["HybridEncoder"])
        self.decoder = DFINETransformer(num_classes=self.num_classes, **model_cfg["DFINETransformer"])

    @staticmethod
    def _resize_linear_out(layer: nn.Linear, output_size: int, del_indices: Optional[set[int]] = None) -> nn.Linear:
        del_indices = del_indices or set()
        old_out, in_features = layer.weight.shape
        keep_indices = [idx for idx in range(old_out) if idx not in del_indices]
        new_layer = nn.Linear(in_features, output_size, bias=layer.bias is not None)
        new_layer = new_layer.to(device=layer.weight.device, dtype=layer.weight.dtype)
        copy_rows = min(len(keep_indices), output_size)
        if copy_rows:
            new_layer.weight.data[:copy_rows] = layer.weight.data[keep_indices[:copy_rows]]
            if layer.bias is not None:
                new_layer.bias.data[:copy_rows] = layer.bias.data[keep_indices[:copy_rows]]
        return new_layer

    @staticmethod
    def _resize_embedding_rows(embedding: nn.Embedding, num_classes: int, del_indices: Optional[set[int]] = None) -> nn.Embedding:
        del_indices = del_indices or set()
        old_padding_idx = embedding.padding_idx
        keep_indices = [idx for idx in range(embedding.num_embeddings) if idx not in del_indices and idx != old_padding_idx]
        new_embedding = nn.Embedding(num_classes + 1, embedding.embedding_dim, padding_idx=num_classes)
        new_embedding = new_embedding.to(device=embedding.weight.device, dtype=embedding.weight.dtype)
        init.normal_(new_embedding.weight)
        copy_rows = min(len(keep_indices), num_classes)
        if copy_rows:
            new_embedding.weight.data[:copy_rows] = embedding.weight.data[keep_indices[:copy_rows]]
        if old_padding_idx is not None:
            new_embedding.weight.data[num_classes] = embedding.weight.data[old_padding_idx]
        return new_embedding

    def resize_output(self, output_size: int, del_indices: Optional[list[int]] = None) -> None:
        """
        Resizes all class-dependent output heads to `output_size`.
        """
        if output_size <= 0:
            raise ValueError(f'output_size must be positive (got {output_size})')

        del_set = set(del_indices or [])
        decoder = self.decoder

        if decoder.query_select_method != "agnostic":
            decoder.enc_score_head = self._resize_linear_out(decoder.enc_score_head, output_size, del_set)

        decoder.dec_score_head = nn.ModuleList(
            [self._resize_linear_out(head, output_size, del_set) for head in decoder.dec_score_head]
        )

        if getattr(decoder, 'denoising_class_embed', None) is not None:
            decoder.denoising_class_embed = self._resize_embedding_rows(decoder.denoising_class_embed, output_size, del_set)

        decoder.num_classes = output_size
        self.num_classes = output_size

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

    def prepare_for_inference(self, config: 'SegmentationInferenceConfig'):
        """
        Configures the model for inference.
        """
        from torchvision.transforms import v2

        self.eval()
        self._inf_config = config

        self._fabric = Fabric(accelerator=self._inf_config.accelerator,
                              devices=self._inf_config.device,
                              precision=self._inf_config.precision)

        self.backbone = self._fabric.to_device(self._fabric._precision.convert_module(self.backbone))
        self.encoder = self._fabric.to_device(self._fabric._precision.convert_module(self.encoder))
        self.decoder = self._fabric.to_device(self._fabric._precision.convert_module(self.decoder))

        _m_dtype = next(self.parameters()).dtype

        self.transforms = v2.Compose([v2.Resize(self.user_metadata['image_size']),
                                      v2.RGB(),
                                      v2.ToImage(),
                                      v2.ToDtype(_m_dtype, scale=True),
                                      v2.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])
        # invert class_mapping
        self.line_map = {}
        self.region_map = {}
        class_mapping = self.user_metadata['class_mapping']
        for cls, idx in class_mapping['lines'].items():
            # there might be multiple classes mapping to the same index -> pick the first one.
            self.line_map.setdefault(idx, cls)
        for cls, idx in class_mapping['regions'].items():
            self.region_map.setdefault(idx, cls)

    @torch.inference_mode()
    def predict(self, im: 'Image.Image') -> 'Segmentation':
        """
        Runs prediction with the model.

        Args:
            im: Input image.
        """
        import shapely.geometry as geom

        from collections import defaultdict
        from torchvision.ops import box_convert
        from kraken.containers import Segmentation, Region, BBoxLine

        orig_size = self._fabric.to_device(torch.tensor(tuple(im.size * 2)))
        scaled_im = self._fabric.to_device(self.transforms(im).unsqueeze(0))
        outputs = self(scaled_im)
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        boxes = box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy') * orig_size

        # scores from distribution
        scores = logits.sigmoid()
        scores, index = torch.topk(scores.flatten(1), self.user_metadata['num_top_queries'], dim=-1)
        labels = index - index // self.num_classes * self.num_classes
        index = index // self.num_classes
        boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))

        # threshold by class score
        scores = scores.squeeze()
        boxes = boxes.squeeze()
        labels = labels.squeeze()
        mask = scores >= 0.5
        boxes, labels = boxes[mask].cpu(), labels[mask].cpu().tolist()

        regions = defaultdict(list)
        _shp_regs = {}
        for box, label in zip(boxes, labels):
            if label in self.region_map:
                bbox = box.index_select(0, torch.tensor([0, 1, 0, 3, 2, 3, 2, 1])).round().to(int).view(4, 2)
                region = Region(id=f'_{uuid.uuid4()}', boundary=bbox.tolist(), tags={'type': [{'type': self.region_map[label]}]})
                regions[self.region_map[label]].append(region)
                _shp_regs[region.id] = geom.Polygon(region.boundary)

        lines = []
        for box, label in zip(boxes, labels):
            if label in self.line_map:
                line_ls = geom.Polygon(box.tolist()).centroid
                regs = []
                for reg_id, reg in _shp_regs.items():
                    if reg.contains(line_ls):
                        regs.append(reg_id)
                lines.append(BBoxLine(id=f'_{uuid.uuid4()}',
                                      bbox=box.tolist(),
                                      tags={'type': [{'type': self.line_map[label]}]},
                                      regions=regs))

        return Segmentation(text_direction=self._inf_config.text_direction,
                            imagename=getattr(im, 'filename', None),
                            type='bbox',
                            lines=lines,
                            regions=regions,
                            script_detection=False,
                            line_orders=[])
