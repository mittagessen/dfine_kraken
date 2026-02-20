#
# Copyright 2025 Benjamin Kiessling
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
dfine.cli.test
~~~~~~~~~~~~~~

Command line driver for object detection model evaluation
"""
import click
import logging

from collections import Counter
from threadpoolctl import threadpool_limits

from .util import _expand_gt, _validate_manifests, message, _create_class_map

logging.captureWarnings(True)
logger = logging.getLogger('dfine')

# suppress worker seeding message
logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)


def _collect_observed_test_classes(test_data):
    """
    Collects raw class names seen in test data before class-map filtering.
    """
    observed = {'lines': Counter(), 'regions': Counter()}
    for sample in test_data or []:
        doc = sample.get('doc') if isinstance(sample, dict) else sample
        if doc is None:
            continue
        for line in getattr(doc, 'lines', []):
            tags = line.tags
            if isinstance(tags, dict):
                line_type = tags.get('type', 'default')
            else:
                line_type = str(tags) if tags else 'default'
            observed['lines'][line_type] += 1
        for reg_type, regs in getattr(doc, 'regions', {}).items():
            observed['regions'][reg_type] += len(regs)
    return observed


def _build_class_diagnostics(model_mapping, dataset_mapping, dataset_stats, observed):
    """
    Builds rows for a unified class diagnostic table.
    """
    rows = []
    mismatches = []
    for section in ('lines', 'regions'):
        model_cls = model_mapping.get(section, {})
        dataset_cls = dataset_mapping.get(section, {})
        section_stats = dataset_stats.get(section, {})
        observed_stats = observed.get(section, {})
        names = sorted(set(model_cls.keys()) | set(dataset_cls.keys()) | set(observed_stats.keys()))
        for name in names:
            model_idx = model_cls.get(name)
            dataset_idx = dataset_cls.get(name)
            observed_count = int(observed_stats.get(name, 0))
            effective_count = int(section_stats.get(name, 0))

            if model_idx is not None and dataset_idx is not None:
                status = 'ok' if model_idx == dataset_idx else 'index mismatch'
            elif dataset_idx is not None:
                status = 'missing in model mapping'
            elif model_idx is not None:
                status = 'missing in dataset mapping'
            elif observed_count > 0:
                status = 'unknown in test data'
            else:
                status = 'unknown'

            if observed_count > 0 and dataset_idx is None and model_idx is not None:
                status = 'ignored by dataset mapping'

            row = {'category': section,
                   'class_name': name,
                   'model_idx': '-' if model_idx is None else str(model_idx),
                   'dataset_idx': '-' if dataset_idx is None else str(dataset_idx),
                   'observed': str(observed_count),
                   'effective': str(effective_count),
                   'status': status}
            rows.append(row)
            if status != 'ok':
                mismatches.append(row)
    return rows, mismatches


@click.command('test')
@click.pass_context
@click.option('-m', '--model', type=click.Path(exists=True, readable=True),
              required=True, help='Model to evaluate')
@click.option('-e',
              '--test-data',
              'test_data',
              multiple=True,
              callback=_validate_manifests,
              type=click.File(mode='r', lazy=True),
              help='File(s) with paths to test data.')
@click.option('-f',
              '--format-type',
              type=click.Choice(['xml', 'alto', 'page']),
              help='Sets the test data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both bounding boxes and a '
              'link to source images.')
@click.option('-B', '--batch-size', type=int, help='batch sample size')
@click.option('-is', '--image-size', type=(int, int), help='Network input image size.')
@click.option('--test-class-mapping-mode',
              type=click.Choice(['full', 'canonical', 'custom']),
              default='full',
              show_default=True,
              help='Class mapping mode for test dataset. `full` uses the '
                   'many-to-one mapping from the training checkpoint (falls '
                   'back to `canonical` for weights files), `canonical` uses '
                   'the one-to-one mapping, `custom` uses user-provided mappings.')
@click.option('--line-class-mapping', type=click.UNPROCESSED, hidden=True)
@click.option('--region-class-mapping', type=click.UNPROCESSED, hidden=True)
@click.argument('test_set', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def test(ctx, **kwargs):
    """
    Evaluate an object detection model on a test set.
    """
    params = ctx.meta.copy()
    params.update(ctx.params)
    model = params.pop('model')
    if not model:
        raise click.UsageError('No model to evaluate given.')

    test_data = params.pop('test_data', [])
    test_set = list(params.pop('test_set', []))

    # merge test_data into test_set list
    if test_data:
        test_set.extend(test_data)

    params['test_data'] = test_set

    if len(test_set) == 0:
        raise click.UsageError('No test data was provided. Use `-e` or the `test_set` argument.')

    # parse custom class mappings
    if isinstance(line_cls_map := params.get('line_class_mapping'), list):
        params['line_class_mapping'] = _create_class_map(line_cls_map)
    if isinstance(region_cls_map := params.get('region_class_mapping'), list):
        params['region_class_mapping'] = _create_class_map(region_cls_map)

    import torch

    from dfine.configs import DFINESegmentationTrainingConfig, DFINESegmentationTestDataConfig
    from dfine.model import DFINESegmentationDataModule, DFINESegmentationModel

    from kraken.train.utils import KrakenTrainer
    from rich.console import Console
    from rich.table import Table

    torch.set_float32_matmul_precision('high')

    trainer = KrakenTrainer(accelerator=ctx.meta['accelerator'],
                            devices=ctx.meta['device'],
                            precision=ctx.meta['precision'],
                            enable_progress_bar=True if not ctx.meta['verbose'] else False,
                            deterministic=ctx.meta['deterministic'],
                            enable_model_summary=False,
                            num_sanity_val_steps=0)

    m_config = DFINESegmentationTrainingConfig(**params)
    dm_config = DFINESegmentationTestDataConfig(**params)
    data_module = DFINESegmentationDataModule(dm_config)

    with trainer.init_module(empty_init=False):
        message(f'Loading from {model}.')
        if model.endswith('ckpt'):
            model = DFINESegmentationModel.load_from_checkpoint(model, config=m_config, weights_only=False)
        else:
            model = DFINESegmentationModel.load_from_weights(model, config=m_config)

    console = Console()

    # initialize test data/model explicitly so diagnostics are based on effective
    # test mapping before running evaluation
    class _DMProxy:
        pass

    class _ModelProxy:
        pass

    dm_proxy = _DMProxy()
    dm_proxy.lightning_module = model
    data_module.trainer = dm_proxy
    data_module.setup('test')

    model_proxy = _ModelProxy()
    model_proxy.datamodule = data_module
    model.trainer = model_proxy
    model.setup('test')

    # class mapping diagnostics
    effective_mapping = data_module.test_set.dataset.class_mapping
    effective_stats = data_module.test_set.dataset.class_stats
    observed = _collect_observed_test_classes(data_module.test_data)
    mode = params.get('test_class_mapping_mode', 'full')
    if mode == 'full' and hasattr(model, '_full_class_mapping'):
        model_mapping = model._full_class_mapping
        model_mapping_src = 'full'
    else:
        model_mapping = model.net.user_metadata['class_mapping']
        model_mapping_src = 'canonical'

    diag_rows, mismatch_rows = _build_class_diagnostics(model_mapping,
                                                        effective_mapping,
                                                        effective_stats,
                                                        observed)
    diag = Table(title=f'Class Mapping Diagnostics (model={model_mapping_src}, dataset=effective)')
    diag.add_column('Category')
    diag.add_column('Class Name')
    diag.add_column('Model Idx')
    diag.add_column('Dataset Idx')
    diag.add_column('Observed')
    diag.add_column('Effective')
    diag.add_column('Status')
    for row in diag_rows:
        diag.add_row(row['category'],
                     row['class_name'],
                     row['model_idx'],
                     row['dataset_idx'],
                     row['observed'],
                     row['effective'],
                     row['status'])
    console.print(diag)

    if mismatch_rows:
        mismatch = Table('Category', 'Class Name', 'Model Idx', 'Dataset Idx', 'Observed', 'Effective', 'Mismatch')
        for row in mismatch_rows:
            mismatch.add_row(row['category'],
                             row['class_name'],
                             row['model_idx'],
                             row['dataset_idx'],
                             row['observed'],
                             row['effective'],
                             row['status'])
        console.print(mismatch)

    with threadpool_limits(limits=ctx.meta['num_threads']):
        test_metrics = trainer.test(model, data_module)

    # overall results table
    overall = Table(title='Overall Detection Metrics')
    overall.add_column('mAP@50', justify='right')
    overall.add_column('mAP@50:95', justify='right')
    overall.add_column('Precision', justify='right')
    overall.add_column('Recall', justify='right')
    overall.add_column('F1', justify='right')
    overall.add_row(f'{test_metrics.map_50:.4f}',
                    f'{test_metrics.map_50_95:.4f}',
                    f'{test_metrics.precision:.4f}',
                    f'{test_metrics.recall:.4f}',
                    f'{test_metrics.f1:.4f}')
    console.print(overall)

    # per-class results table (no mAP@50, torchmetrics does not provide it)
    if test_metrics.per_class_metrics:
        per_class = Table(title='Per-Class Detection Metrics')
        per_class.add_column('Class')
        per_class.add_column('mAP@50:95', justify='right')
        per_class.add_column('Precision', justify='right')
        per_class.add_column('Recall', justify='right')
        per_class.add_column('F1', justify='right')
        for cls_name, m in sorted(test_metrics.per_class_metrics.items()):
            per_class.add_row(cls_name,
                              f'{m["map_50_95"]:.4f}',
                              f'{m["precision"]:.4f}',
                              f'{m["recall"]:.4f}',
                              f'{m["f1"]:.4f}')
        console.print(per_class)

    message(f'mAP@50: {test_metrics.map_50:.4f}  mAP@50:95: {test_metrics.map_50_95:.4f}')
