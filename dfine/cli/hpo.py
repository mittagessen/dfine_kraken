"""
dfine.cli.hpo
~~~~~~~~~~~~~

Command line driver for Optuna hyperparameter optimization.
"""
import gc
import os
import click
import logging

from threadpoolctl import threadpool_limits

from .util import _expand_gt, _validate_manifests, _create_class_map

logging.captureWarnings(True)
logger = logging.getLogger('dfine')

# suppress worker seeding message
logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)


@click.command('hpo')
@click.option('-N', '--epochs', type=int, default=None,
              help='Override epochs per trial.')
@click.option('--n-trials', type=int, default=50,
              help='Number of Optuna trials.')
@click.option('--output-dir', required=True, type=click.Path(),
              help='Directory for SQLite study DB.')
@click.option('--study-name', default='dfine_hpo',
              help='Optuna study name.')
@click.option('--model-variant',
              type=click.Choice(['nano', 'small', 'medium', 'large', 'extra_large']),
              help='D-FINE model variant.')
@click.option('--num-top-queries', type=int,
              help='Number of top query predictions used for validation/inference.')
@click.option('--gradient-clip-val', type=float,
              help='Gradient clip value.')
@click.option('-p', '--partition', type=float,
              help='Ground truth data partition ratio between train/validation set.')
@click.option('-t', '--training-data', '--training-files', 'training_data', default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data.')
@click.option('-e', '--evaluation-data', '--evaluation-files', 'evaluation_data', default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter.')
@click.option('-f', '--format-type',
              type=click.Choice(['xml', 'alto', 'page']),
              help='Sets the training data format.')
@click.option('-is', '--image-size', type=(int, int), help='Network input image size.')
@click.option('--line-class-mapping', type=click.UNPROCESSED, hidden=True)
@click.option('--region-class-mapping', type=click.UNPROCESSED, hidden=True)
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
@click.pass_context
def hpo(ctx, **kwargs):
    """
    Runs Optuna hyperparameter optimization for D-FINE segmentation models.

    Explores learning rate, schedule, warmup, and weight decay. Results
    persist in SQLite for resumability.
    """
    import torch
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback

    from dfine.configs import (DFINESegmentationTrainingConfig,
                               DFINESegmentationTrainingDataConfig)
    from dfine.model import DFINESegmentationDataModule, DFINESegmentationModel
    from kraken.train.utils import KrakenTrainer

    params = ctx.params.copy()
    params.update(ctx.meta)

    n_trials = params.pop('n_trials')
    output_dir = params.pop('output_dir')
    study_name = params.pop('study_name')

    training_data = params.pop('training_data', [])
    ground_truth = list(params.pop('ground_truth', []))

    if training_data:
        ground_truth.extend(training_data)

    params['training_data'] = ground_truth

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the hpo command. Use `-t` or the `ground_truth` argument.')

    # parse class mappings
    if isinstance(line_cls_map := params.get('line_class_mapping'), list):
        params['line_class_mapping'] = _create_class_map(line_cls_map)
    if isinstance(region_cls_map := params.get('region_class_mapping'), list):
        params['region_class_mapping'] = _create_class_map(region_cls_map)

    # disable automatic partition when given evaluation set explicitly
    if params.get('evaluation_data'):
        params['partition'] = 1

    epochs = params.pop('epochs', None) or params.get('epochs', 50)

    torch.set_float32_matmul_precision('high')
    os.makedirs(output_dir, exist_ok=True)

    # Build data module once (XML parsing is expensive)
    dm_config = DFINESegmentationTrainingDataConfig(**params)
    logger.info('Parsing training data and building data module...')
    data_module = DFINESegmentationDataModule(dm_config)
    logger.info(f'Data module ready: {len(data_module.train_set)} train, '
                f'{len(data_module.val_set)} val samples')

    def objective(trial):
        """Single Optuna trial: suggest hyperparams, train, return mAP@50."""
        lrate = trial.suggest_float('lrate', 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
        warmup = trial.suggest_int('warmup', 0, 2000, step=50)
        schedule = trial.suggest_categorical('schedule',
                                             ['constant', 'cosine', 'step',
                                              'reduceonplateau', '1cycle'])

        sched_kw = {}
        if schedule == 'cosine':
            sched_kw['cos_t_max'] = trial.suggest_int('cos_t_max', 3, max(epochs, 3))
            sched_kw['cos_min_lr'] = trial.suggest_float('cos_min_lr', 1e-7,
                                                          max(lrate / 10, 1.5e-7),
                                                          log=True)
        elif schedule == 'step':
            sched_kw['step_size'] = trial.suggest_int('step_size', 2, 10)
            sched_kw['gamma'] = trial.suggest_float('gamma', 0.1, 0.9)
        elif schedule == 'reduceonplateau':
            sched_kw['rop_factor'] = trial.suggest_float('rop_factor', 0.1, 0.5)
            sched_kw['rop_patience'] = trial.suggest_int('rop_patience', 2, 10)

        m_config = DFINESegmentationTrainingConfig(
            model_variant=params.get('model_variant', 'large'),
            num_top_queries=params.get('num_top_queries', 300),
            quit='fixed',
            epochs=epochs,
            lrate=lrate,
            weight_decay=weight_decay,
            warmup=warmup,
            schedule=schedule,
            **sched_kw,
        )

        trainer = KrakenTrainer(
            accelerator=params.get('accelerator'),
            devices=params.get('device'),
            precision=params.get('precision'),
            max_epochs=epochs,
            min_epochs=0,
            enable_progress_bar=False,
            enable_summary=False,
            enable_checkpointing=False,
            gradient_clip_val=params.get('gradient_clip_val', 1.0),
            num_sanity_val_steps=0,
            use_distributed_sampler=False,
            pl_logger=None,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_metric')],
        )

        with trainer.init_module(empty_init=True):
            model = DFINESegmentationModel(m_config)

        try:
            with threadpool_limits(limits=params.get('num_threads', 1)):
                trainer.fit(model, data_module)
            val_metric = trainer.callback_metrics.get('val_metric')
            return val_metric.item() if val_metric is not None else float('nan')
        finally:
            del model
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Create study with SQLite for resumability
    storage = f'sqlite:///{os.path.abspath(output_dir)}/hpo_study.db'
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=3),
    )

    study.optimize(objective, n_trials=n_trials)

    # Print results
    click.echo(f'\nCompleted {len(study.trials)} trials.')
    click.echo(f'\nBest trial (#{study.best_trial.number}):')
    click.echo(f'  mAP@50: {study.best_trial.value:.4f}')
    click.echo('  Params:')
    for key, value in study.best_trial.params.items():
        click.echo(f'    {key}: {value}')

    completed = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=True,
    )
    click.echo('\nTop-5 trials:')
    click.echo(f'  {"#":<5} {"mAP@50":<10} {"lrate":<12} {"weight_decay":<14} '
               f'{"schedule":<16} {"warmup":<8}')
    click.echo(f'  {"-" * 65}')
    for t in completed[:5]:
        click.echo(f'  {t.number:<5} {t.value:<10.4f} {t.params["lrate"]:<12.2e} '
                   f'{t.params["weight_decay"]:<14.2e} {t.params["schedule"]:<16} '
                   f'{t.params["warmup"]:<8}')
