#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
dfine.cli.infer
~~~~~~~~~~~~~~~

Command line driver for running inference on images and producing ALTO XML output.
"""
import glob
import logging
import os

import click

from .util import message

logging.captureWarnings(True)
logger = logging.getLogger('dfine')

logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)


@click.command('infer')
@click.pass_context
@click.option('-m', '--model',
              type=click.Path(exists=True, readable=True),
              required=True,
              help='Path to the model weights (.safetensors or .ckpt).')
@click.option('-o', '--output-dir',
              'output_dir',
              type=click.Path(file_okay=False, writable=True),
              default='.',
              show_default=True,
              help='Directory where ALTO XML files will be written.')
@click.option('--nms-iou',
              'nms_iou',
              type=float,
              default=None,
              show_default=True,
              help='IoU threshold for NMS post-processing. If not set, no NMS is applied.')
@click.option('--score-threshold',
              'score_threshold',
              type=float,
              default=0.5,
              show_default=True,
              help='Minimum confidence score to keep a detection.')
@click.option('-d', '--device',
              default=None,
              help='Device to use (cpu, cuda:0, ...). Defaults to the global --device option.')
@click.argument('images', nargs=-1, required=True)
def infer(ctx, model, output_dir, nms_iou, score_threshold, device, images):
    """
    Run inference on a set of images and write one ALTO XML file per image.

    IMAGES can be a glob pattern or a list of image paths, e.g.:

        dfine infer -m model.safetensors -o results/ 'images/*.jpg'

    """
    import torch
    from PIL import Image as PILImage
    from kraken import serialization
    from kraken.configs import SegmentationInferenceConfig
    from kraken.lib.util import open_image

    from dfine.configs import DFINESegmentationTrainingConfig
    from dfine.model import DFINESegmentationModel

    torch.set_float32_matmul_precision('high')

    # resolve glob patterns
    image_paths = []
    for pattern in images:
        expanded = glob.glob(pattern)
        if expanded:
            image_paths.extend(sorted(expanded))
        elif os.path.exists(pattern):
            image_paths.append(pattern)
        else:
            logger.warning(f'Pattern {pattern!r} matched no files, skipping.')

    if not image_paths:
        raise click.UsageError('No images found. Check your glob pattern.')

    os.makedirs(output_dir, exist_ok=True)

    # resolve device: CLI flag overrides global context
    accelerator = ctx.meta.get('accelerator', 'cpu')
    ctx_device = ctx.meta.get('device', 'auto')
    if device is not None:
        from dfine.cli.util import to_ptl_device
        accelerator, ctx_device = to_ptl_device(device)

    precision = ctx.meta.get('precision', '32-true')

    # build a minimal inference config
    inf_config = SegmentationInferenceConfig(accelerator=accelerator,
                                             device=ctx_device,
                                             precision=precision,
                                             nms_iou=nms_iou)

    # load model
    message(f'Loading model from {model}.')
    m_config = DFINESegmentationTrainingConfig()
    if model.endswith('.ckpt'):
        seg_model = DFINESegmentationModel.load_from_checkpoint(model,
                                                                config=m_config,
                                                                weights_only=False)
    else:
        seg_model = DFINESegmentationModel.load_from_weights(model, config=m_config)

    net = seg_model.net
    net.prepare_for_inference(inf_config, nms_iou=nms_iou)

    message(f'Running inference on {len(image_paths)} image(s) '
            f'(nms_iou={nms_iou}, score_threshold={score_threshold}).')

    for img_path in image_paths:
        try:
            im = open_image(img_path)
        except IOError as e:
            logger.error(f'Could not open {img_path}: {e}')
            continue

        try:
            segmentation = net.predict(im)
        except Exception as e:
            logger.error(f'Inference failed on {img_path}: {e}')
            continue

        # build output path: same basename, .xml extension
        basename = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, basename + '.xml')

        alto_str = serialization.serialize(segmentation,
                                           image_size=im.size,
                                           template='alto',
                                           template_source='native')

        with open(out_path, 'w', encoding='utf-8') as fp:
            fp.write(alto_str)

        message(f'  {img_path} -> {out_path}')

    message('Done.')
