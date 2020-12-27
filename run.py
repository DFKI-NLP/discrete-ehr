import logging
import shlex
import sys
from functools import partial
from glob import glob
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib.pyplot as plt
import torch
import utils
import wandb
from dataloader.data import MIMICDataset, TabularFeature, get_tables
from dataloader.labels import Label, get_labels
from dataloader.utils import get_vocab
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, EventEnum, Events
from ignite.handlers import (Checkpoint, ModelCheckpoint, Timer,
                             global_step_from_engine)
from ignite.metrics import Average
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import MomentumContrastiveModel


def train(model: torch.nn.Module, device,
          train_loader: DataLoader, val_loader: DataLoader,
          tables: List[TabularFeature], labels: Dict[str, Label],
          n_epochs: int, tboardwriter=None, resume=False, finetune=False,
          init_eval=False, drop_checkpoint_embeddings=False,
          clip_grad=10,
          **params):
    tunable_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(tunable_params, lr=params['lr'], weight_decay=params['weight_decay'])
    optimizer.zero_grad()

    update_at = (params['batch_update'] / params['batch_size'])

    save_dir = wandb.run.dir

    def process_function(engine, batch):
        model.train()
        # engine.state.iteration starts at 1
        if (engine.state.iteration - 1) % update_at == 0:
            optimizer.zero_grad()

        x, y_true, _ = utils.prepare_batch(batch, device)
        preds, output = model(*x.values())
        output.update({"y_pred": preds,
                       "y_true": y_true})

        losses = {}
        reported_losses = {}
        for label in labels.values():
            try:
                losses[label.task] = label.loss(output) / update_at
                reported_losses[label.task] = losses[label.task].item()
            except:
                reported_losses[label.task] = 0.
                continue

        if losses:
            loss: torch.Tensor = sum(losses.values())
            loss.backward()
            loss = loss.item()
        else:
            loss = 0.

        output.update({"loss": loss,
                       "losses": {k: v for k, v in reported_losses.items()}})
        engine.state.batch_loss.append(output['loss'])
        engine.state.batch_losses.append(output['losses'])

        if np.isnan(list(output['losses'].values())).any():
            # breakpoint()
            pass

        if engine.state.iteration % update_at == 0:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)

            if isinstance(model, MomentumContrastiveModel):
                model._momentum_update_key_encoder()  # update the key encoder

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            engine.fire_event(CustomEvents.BATCH_UPDATE)

        return output

    def eval_function(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y_true, extra = utils.prepare_batch(batch, device)

            preds, output = model(*x.values())
            output.update({"y_pred": preds,
                           "y_true": y_true})

            losses = {}
            reported_losses = {}
            for label in labels.values():
                try:
                    losses[label.task] = label.loss(output) / update_at
                    reported_losses[label.task] = losses[label.task].item()
                except:
                    reported_losses[label.task] = 0.
                    continue

            if losses:
                loss = sum(losses.values())
                loss = loss.item()
            else:
                loss = 0.

            output.update({"loss": loss,
                           "losses": {k: v for k, v in reported_losses.items()}})

            output.update(extra)
            return output

    class CustomEvents(EventEnum):
        BATCH_UPDATE = "batch_update"

    trainer = Engine(process_function)
    trainer.register_events(*CustomEvents)
    validation_evaluator = Engine(eval_function)

    scheduler = None
    if params['cyclical_lr']:
        # CANNOT RESUME
        base_lr, max_lr = params['cyclical_lr']
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr,
                                                      step_size_up=len(train_loader)/params['batch_update'] * 2,
                                                      mode='triangular2', cycle_momentum=False)

    def attach_metrics(engine, mode):
        Average(output_transform=lambda x: x['loss']).attach(engine, 'loss')
        earlystop_criterium = []
        for task, label in labels.items():
            _earlystop_criterium = label.attach_metrics(engine, mode, **params)
            earlystop_criterium.append(_earlystop_criterium)
        sum(earlystop_criterium).attach(engine, 'earlystop_criterium')

    attach_metrics(trainer, 'TRAIN')
    attach_metrics(validation_evaluator, 'EVAL')

    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(trainer, output_transform=lambda x: x['losses'])
    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(validation_evaluator, output_transform=lambda x: x['losses'])

    @trainer.on(Events.EPOCH_STARTED)
    def init_batch_loss(engine):
        engine.state.batch_loss = []
        engine.state.batch_losses = []
        engine.state.epoch_loss = []

    @trainer.on(CustomEvents.BATCH_UPDATE)
    def log_batch_loss(engine):
        batch_loss = np.sum(engine.state.batch_loss)
        engine.state.epoch_loss.append(batch_loss)
        tboardwriter.add_scalar('batch/loss', batch_loss, engine.state.iteration // update_at)
        # wandb.log({'batch/loss': batch_loss}, step=engine.state.iteration // update_at)
        for task, _ in labels.items():
            tboardwriter.add_scalar(f'batch/loss_{task}', np.sum([l[task] for l in engine.state.batch_losses]), engine.state.iteration // update_at)

        engine.state.batch_loss = []
        engine.state.batch_losses = []

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        if hasattr(engine.state, 'epoch_loss'):
            engine.state.metrics['loss'] = np.mean(engine.state.epoch_loss)
            engine.state.epoch_loss = []

        metrics = engine.state.metrics
        scalar_metrics = [(k, v) for k, v in metrics.items() if 'plot' not in k]
        logging.info(
            f"Training Results - Epoch: {trainer.state.epoch},  {scalar_metrics}")
        logs = {}

        for metric_name, value in metrics.items():
            if 'line_plot' in metric_name:
                fig = plt.figure(figsize=(6, 6), dpi=72, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(value)
                plt.ylim(0, 1)
                value = fig
                tboardwriter.add_figure(f'train/{metric_name}', value, trainer.state.epoch)
            else:
                tboardwriter.add_scalar(f'train/{metric_name}', value, trainer.state.epoch)
            logs[f'train/{metric_name}'] = value
        wandb.log(logs, step=trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        for name, param in model.named_parameters():
            if param.requires_grad:
                try:
                    tboardwriter.add_histogram(name, param.data.detach().clone().cpu().numpy(), trainer.state.epoch)
                except:
                    pass

        validation_evaluator.run(val_loader)

        metrics = validation_evaluator.state.metrics
        scalar_metrics = [(k, v) for k, v in metrics.items() if 'plot' not in k]
        logging.info(
            f"Validation Results - Epoch: {trainer.state.epoch},  {scalar_metrics}")
        logs = {}
        for metric_name, value in metrics.items():
            if 'line_plot' in metric_name:
                fig = plt.figure(figsize=(6, 6), dpi=72, facecolor='w', edgecolor='k')
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(value)
                plt.ylim(0, 1)
                value = fig
                tboardwriter.add_figure(f'val/{metric_name}', value, trainer.state.epoch)
            else:
                tboardwriter.add_scalar(f'val/{metric_name}', value, trainer.state.epoch)
            logs[f'val/{metric_name}'] = value
        logs.update({"activations/" + k: wandb.Histogram(np_histogram=(v, act_hist_handler.bins[k])) for k, v in act_hist_handler.histograms.items()})

        wandb.log(logs, step=trainer.state.epoch)

    @trainer.on(Events.STARTED)
    def init_trainer(engine):
        if init_eval:
            validation_evaluator.run(train_loader)
            log_training_results(validation_evaluator)
            log_validation_results(engine)

    epoch_timer = Timer(average=True)
    epoch_timer.attach(trainer,
                       start=Events.EPOCH_STARTED,
                       resume=Events.ITERATION_STARTED,
                       pause=Events.ITERATION_COMPLETED,
                       step=Events.EPOCH_COMPLETED)
    batch_timer = Timer(average=True)
    batch_timer.attach(trainer,
                       start=Events.EPOCH_STARTED,
                       resume=Events.ITERATION_STARTED,
                       pause=Events.ITERATION_COMPLETED,
                       step=CustomEvents.BATCH_UPDATE)
    eval_timer = Timer(average=True)
    eval_timer.attach(validation_evaluator,
                      start=Events.EPOCH_STARTED,
                      resume=Events.ITERATION_STARTED,
                      pause=Events.ITERATION_COMPLETED,
                      step=Events.ITERATION_COMPLETED)

    act_hist_handler = utils.ActivationHandler(n_bins=20)
    validation_evaluator.add_event_handler(Events.ITERATION_COMPLETED, act_hist_handler)

    # EARLY STOP
    def score_function(engine):
        return engine.state.metrics['earlystop_criterium']
    early_stop = utils.EarlyStopping(patience=params['patience'],
                                     score_function=score_function,
                                     trainer=trainer)
    validation_evaluator.add_event_handler(Events.COMPLETED, early_stop)

    # PERSISTENCE
    PERSISTANCE_DICT = {'trainer': trainer, 'evaluator': validation_evaluator,
                        'model': model, 'optimizer': optimizer,
                        }

    if scheduler is not None:
        PERSISTANCE_DICT['scheduler'] = scheduler

    def load_latest_checkpoint(glob_str, wandb_id=wandb.run.id):
        model_paths = glob(f'wandb/*-{wandb_id}/**/{glob_str}')
        latest_model_path = sorted(model_paths)[-1]
        logging.info(f'LOAD LATEST CHECKPOINT AT {latest_model_path}')

        with open(latest_model_path, 'rb') as checkpoint_file:
            checkpoint = torch.load(checkpoint_file)
        return checkpoint, latest_model_path

    if resume:
        checkpoint, _ = load_latest_checkpoint('last*.pt', resume)
        Checkpoint.load_objects(to_load=PERSISTANCE_DICT, checkpoint=checkpoint,
                                strict=False)
    elif finetune:
        logging.info('FINETUNING')
        checkpoint, loaded_path = load_latest_checkpoint('best*', wandb_id=finetune)
        if drop_checkpoint_embeddings:
            del checkpoint['model']['timestep_encoder.event_encoder.encoder.weight']
        Checkpoint.load_objects(to_load={'model': model}, checkpoint=checkpoint,
                                strict=False)
        wandb.config.update({'finetune': loaded_path}, allow_val_change=True)
    else:
        torch.save(model.state_dict(), f'{save_dir}/init_checkpoint_0.pt')

    best_model_saver = ModelCheckpoint(save_dir, 'best', score_function=score_function,
                                       require_empty=False, n_saved=1,
                                       global_step_transform=global_step_from_engine(trainer))
    validation_evaluator.add_event_handler(Events.COMPLETED, best_model_saver, PERSISTANCE_DICT)
    model_saver = ModelCheckpoint(save_dir, 'last',
                                  require_empty=False, n_saved=1,
                                  global_step_transform=lambda engine, _: engine.state.epoch)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, model_saver, PERSISTANCE_DICT)

    # Run training only if we are not resuming, if resuming make sure the n_epoch is larger than trainer epoch.
    trainer.run(train_loader, max_epochs=n_epochs)

    metrics = {}
    metrics.update({f'tr_{k}': v for k, v in trainer.state.metrics.items() if 'plot' not in k})
    metrics.update({f'best_val_{k}': v for k, v in validation_evaluator.state.best_metrics.items() if 'plot' not in k})

    training_summary = {'time_epoch': epoch_timer.value(),
                        'time_batch': batch_timer.value(),
                        'eval_timer': eval_timer.value(),
                        'stop_epoch': getattr(trainer.state, 'stop_epoch', trainer.state.epoch)}
    wandb.run.summary.update(training_summary)
    metrics.update(training_summary)

    return metrics


def run(n_epochs, name, data_path, dev, **params):
    DEVICE = 'cuda' if not dev else 'cpu'

    torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_path = f'runs/logs/{name}.log'
    log_level = 'DEBUG' if dev else 'INFO'
    utils.setup_logger(log_path, log_level)
    logging.info("---NEW TRAINING---")
    logging.info(sys.argv[0] + ' ' + ' '.join([shlex.quote(s) for s in sys.argv[1:]]))

    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    logging.info(f'TRAIN: {params}')
    tboardwriter = SummaryWriter('runs/' + f'{Path(wandb.run.dir).name}_{wandb.run.name}')

    labels = get_labels(DEVICE, **params)

    joint_vocab = get_vocab(**params)
    logging.info(f'Vocab size: {len(joint_vocab)}')
    tables = get_tables(vocab=joint_vocab,
                        load=True,
                        event_class=utils.load_class(params['eventcls']),
                        **params)
    # value_vocab = build_value_vocab([table for table in tables if isinstance(table, TabularFeature)])
    # for table in [table for table in tables if isinstance(table, TabularFeature)]:
        # table.value_vocab = value_vocab
    if params['datalimit']:
        data_limit = int(29250 * params['datalimit'])
    else:
        data_limit = None

    if not params['random_split']:
        train_set = MIMICDataset(data_path, 'train', datalist_file='train_listfile.csv', mode='TRAIN',
                                 tables=tables, labels=labels,
                                 limit='^(1111|222)' if dev else data_limit,
                                 numericalize=True)
        val_set = MIMICDataset(data_path, 'train', datalist_file='val_listfile.csv', mode='EVAL',
                               tables=tables, labels=labels,
                               limit='^(11).*' if dev else None,
                               numericalize=True)
    else:
        logging.warning('DATA: RANDOM SPLIT')
        train_set = MIMICDataset(data_path, 'train', datalist_file='train/listfile.csv', mode='TRAIN',
                                 tables=tables, labels=labels,
                                 limit='^(11)' if dev else None,
                                 numericalize=True)
        train_set, val_set = train_set.split()

    # TEST
    assert (list(train_set[0]['inputs'].values())[1][:,:,0] != 0).any()
    assert (list(train_set[0]['inputs'].values())[1][:,:,1] != 0).any()
    if params['sep_bin']:
        assert not [v for v in joint_vocab.itos if v.endswith('=1')]
    else:
        assert [v for v in joint_vocab.itos if v.endswith('=1')]

    logging.info(f'Training on {len(train_set)}   Validating on {len(val_set)}')

    sampler_class = utils.load_class(params['sampler'])
    sampler = sampler_class(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=params['batch_size'],
                                               collate_fn=partial(utils.min_batch,
                                                                  tables=tables,
                                                                  labels=labels,
                                                                  limit=params['step_limit'],
                                                                  event_limit=params['event_limit']),
                                               sampler=sampler,
                                               num_workers=params['num_workers'] if not dev else 0, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1,
                                             collate_fn=partial(utils.pad_batch,
                                                                tables=tables,
                                                                labels=labels,
                                                                limit=None),
                                             shuffle=False, num_workers=params['num_workers'] if not dev else 0, pin_memory=True)

    model, device = utils.create_model_on_gpu(params, DEVICE, tables=tables,
                                              joint_vocab=joint_vocab,
                                              # value_vocab=value_vocab
                                              )

    logging.info(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total params: {total_params}, trainable params: {trainable_params}')
    param_summary = {'total_params': total_params,
                     'trainable_params': trainable_params}

    wandb.watch(model, log="all")
    wandb.config.update(param_summary, allow_val_change=True)

    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    # torch.save(model.state_dict(), wandb.run.dir + '/init_checkpoint.pt')

    metrics = train(model, device, train_loader, val_loader,
                    tables, labels, n_epochs, tboardwriter=tboardwriter,
                    **params)

    tboardwriter.add_hparams({k: v for k, v in params.items() if k not in ['tasks', 'input_tables', 'prediction_steps', 'cyclical_lr', 'datalimit']},
                             {f'hp/{metric_name}': metric for metric_name, metric in metrics.items()
                             if metric_name in ['best_val_decompensation_ap',
                                                'best_val_in_hospital_mortality_ap',
                                                'best_val_phenotyping_ap_macro',
                                                'best_val_length_of_stay_classification_kappa',
                                                'best_val_length_of_stay_classification_mad',
                                                'best_val_contrastive_acc']})
    tboardwriter.close()
