import csv
import gzip
import importlib
import logging
import math
import os
import sys
from collections import defaultdict
from glob import glob
from os import makedirs
from pathlib import Path
from typing import Dict, List, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from colorlog import ColoredFormatter
from ignite.engine import Engine
from sklearn.metrics import auc

import wandb
from dataloader.data import DemographicsFeature, TabularFeature, JointTabularFeature


class EarlyStopping(object):
    """EarlyStopping handler can be used to stop the training if no improvement after a given number of events.

    Args:
        patience (int):
            Number of events to wait if no improvement and then stop the training.
        score_function (callable):
            It should be a function taking a single argument, an :class:`~ignite.engine.Engine` object,
            and return a score `float`. An improvement is considered if the score is higher.
        trainer (Engine):
            trainer engine to stop the run if no improvement.
        callback (callable):
            You can pass a function to be called everytime an early stopping point is marked using the score_function.

    Examples:

    .. code-block:: python

        from ignite.engine import Engine, Events
        from ignite.handlers import EarlyStopping

        def score_function(engine):
            val_loss = engine.state.metrics['nll']
            return -val_loss

        handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
        # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
        evaluator.add_event_handler(Events.COMPLETED, handler)

    """

    def __init__(self, patience, score_function, trainer):
        if not callable(score_function):
            raise TypeError("Argument score_function should be a function.")

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if not isinstance(trainer, Engine):
            raise TypeError("Argument trainer should be an instance of Engine.")

        self.score_function = score_function
        self.patience = patience
        self.trainer = trainer
        self.counter = 0
        self.best_score = None
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())

    def __call__(self, engine):
        score = self.score_function(engine)

        if self.best_score is None:
            self.best_score = score
            engine.state.best_metrics = {k: v for k, v in engine.state.metrics.items() if 'skip' not in k}
            wandb.run.summary.update({f'best_val_{k}': v for k, v in engine.state.metrics.items() if 'plot' not in k})
            self.trainer.state.stop_epoch = self.trainer.state.epoch
        elif score <= self.best_score:
            self.counter += 1
            self._logger.debug("EarlyStopping: %i / %i" % (self.counter, self.patience))
            if self.counter >= self.patience:
                self._logger.info("EarlyStopping: Stop training")
                self.trainer.terminate()
        else:
            self.best_score = score
            engine.state.best_metrics = {k: v for k, v in engine.state.metrics.items() if 'skip' not in k}
            wandb.run.summary.update({f'best_val_{k}': v for k, v in engine.state.metrics.items() if 'plot' not in k})
            self.trainer.state.stop_epoch = self.trainer.state.epoch
            self.counter = 0


class ActivationHandler:
    def __init__(self, n_bins: int, global_step_transform=None):
        self.n_bins = n_bins
        self.global_step_transform = global_step_transform
        self._reset()

    def _reset(self):
        self.histograms = defaultdict(lambda: np.zeros(self.n_bins))
        self.bins = defaultdict(lambda: np.zeros(self.n_bins + 1))

    def __call__(self, engine):
        if engine.state.iteration == 1:
            self.histograms['timesteps'], self.bins['timesteps'] = np.histogram(torch.cat(engine.state.output['timesteps'], -1).cpu(), self.n_bins)
            self.histograms['patient'], self.bins['patient'] = np.histogram(engine.state.output['patient'].cpu(), self.n_bins)
        elif engine.state.iteration <= len(engine.state.dataloader):
            self.histograms['timesteps'] += np.histogram(torch.cat(engine.state.output['timesteps'], -1).cpu(), self.bins['timesteps'])[0]
            self.histograms['patient'] += np.histogram(engine.state.output['patient'].cpu(), self.bins['patient'])[0]
        else:
            raise RuntimeError


def setup_logger(path, level="INFO"):
    formatter = ColoredFormatter('%(asctime)s|%(funcName)s|%(levelname)s: %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(level)

    makedirs(Path(path).parent, exist_ok=True)
    file_handler = logging.FileHandler(path, mode='a')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)


def create_model_on_gpu(pc, device_name="cuda:0", **kwargs):
    torch.cuda.empty_cache()
    model = load_class(pc['modelcls'])(**pc, **kwargs)
    logging.info(model)
    device = torch.device(device_name)
    return model.to(device), device


def multidim_pad_sequences(sequences, batch_first, padding_value=0):
    max_size = sequences[0].size()
    trailing_dims = max_size[2:]
    max_len = np.max([[s.size(0), s.size(1)] for s in sequences], 0)
    if batch_first:
        out_dims = (len(sequences), *max_len) + trailing_dims
    else:
        out_dims = (*max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length1 = tensor.size(0)
        length2 = tensor.size(1)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length1, :length2, ...] = tensor
        else:
            out_tensor[:length1, i, :length2, ...] = tensor
    return out_tensor


def pad_batch(batch, tables=[], labels={}, limit=None, event_limit=None):
    x = {}

    x['extra'] = {}
    for key in batch[0]['extra'].keys():
        x['extra'][key] = [sample['extra'][key] for sample in batch]

    x['inputs'] = {}
    for table in tables:
        if isinstance(table, TabularFeature) or isinstance(table, JointTabularFeature):
            # N, L, L, C
            x['inputs'][table.table] = multidim_pad_sequences([sample['inputs'][table.table][:limit] for sample in batch], batch_first=True)
            x['inputs'][table.table] = x['inputs'][table.table][:,:,:event_limit]
        elif isinstance(table, DemographicsFeature):
            x['inputs'][table.table] = torch.stack([sample['inputs'][table.table] for sample in batch])

        if x['inputs'][table.table].device.type == 'cuda':
            x['inputs'][table.table] = x['inputs'][table.table].pin_memory()

    x['targets'] = {}
    for key, label in labels.items():
        x['targets'][key] = label.batch(batch)
        if x['targets'][key].device.type == 'cuda':
            x['targets'][key] = x['targets'][key].pin_memory()

    return x


def multidim_shortest_sequences(sequences, batch_first=True, event_limit=None, padding_value=0):
    '''Truncates and pads a list of patient histories. Permutes events.
    sequences: N length sorted sequences list of tensors with shape L1, L2, *.
    returns N, truncated timesteps dimension L1, padded event dimension L2, *
    '''
    trailing_dims = sequences[0].shape[2:]

    # Assume length sorted sequences
    min_length = int(np.mean([s.size(0) for s in sequences]))
    if event_limit:
        max_events = min(event_limit, int(np.mean([s.size(1) for s in sequences])))
    else:
        max_events = int(np.mean([s.size(1) for s in sequences]))

    length1 = min_length
    length2 = max_events

    if batch_first:
        out_dims = (len(sequences), length1, length2, *trailing_dims)
    else:
        out_dims = (*length1, len(sequences), length2, *trailing_dims)

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        if length1 >= tensor.size(0):
            L1 = length1
            repeat_times = math.ceil(length1 / tensor.size(0))
            tensor = tensor.repeat(repeat_times, *([1] * (tensor.ndim - 1)))
        else:
            L1 = length1
        if length2 >= tensor.size(1):
            L2 = tensor.size(1)
        else:
            L2 = length2
        # randomly sample the measures
        L2ix = np.random.permutation(range(L2))
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :L1, :L2, ...] = tensor[:L1, L2ix]
        else:
            out_tensor[:L1, i, :L2, ...] = tensor[:L1, L2ix]
    return out_tensor


def min_batch(batch, tables=[], labels={}, limit=None, event_limit=None):
    x = {}

    # sort batch to make the positive sample first and thus with least padding
    # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
    time_lengths = [list(s['inputs'].values())[-1].size(0) for s in batch]
    batch = [sample for _, sample in sorted(zip(time_lengths, batch),
                                            key=lambda pair: pair[0],
                                            reverse=True)]
    
    x['extra'] = {}
    for key in batch[0]['extra'].keys():
        x['extra'][key] = [sample['extra'][key] for sample in batch]

    x['inputs'] = {}
    for table in tables:
        if isinstance(table, TabularFeature) or isinstance(table, JointTabularFeature):
            # N, L, L, C
            x['inputs'][table.table] = multidim_shortest_sequences([sample['inputs'][table.table][:limit] for sample in batch], batch_first=True, event_limit=event_limit)
        elif isinstance(table, DemographicsFeature):
            x['inputs'][table.table] = torch.stack([sample['inputs'][table.table] for sample in batch])
        
        if x['inputs'][table.table].device.type == 'cuda':
            x['inputs'][table.table] = x['inputs'][table.table].pin_memory()

    x['targets'] = {}
    for key, label in labels.items():
        x['targets'][key] = label.batch(batch)
        if x['targets'][key].device.type == 'cuda':
            x['targets'][key] = x['targets'][key].pin_memory()
        
    return x


def plot_confusion_matrix(cm, label, ax=None, annot=True,fmt='d', square=True, **kwargs):
    """
    Keyword Arguments:
    correct_labels -- These are your true classification categories.
    predict_labels -- These are you predicted classification categories
    label          -- This is a list of string labels corresponding labels

    Returns: Figure
    """

    if not ax:
        fig = plt.figure(figsize=(6, 6), dpi=72, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)

    df = pd.DataFrame(cm, label.classes, label.classes)

    ax = sns.heatmap(df, annot=annot, cmap='Oranges', fmt=fmt, cbar=False, square=square, ax=ax, **kwargs)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True label')
    plt.tight_layout()

    return ax, cm


def plot_pr_curve(precision, recall):
    """
    Keyword Arguments:
    correct_labels -- These are your true classification categories.
    predict_labels -- These are you predicted classification categories
    labels         -- This is a list of values that occur in y_true
    classes        -- This is a list of string labels corresponding labels

    Returns: Figure
    """
    aucpr = auc(recall, precision)

    fig = plt.figure(figsize=(6, 6), dpi=72, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(recall, precision)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'AUCPR={aucpr}')
    fig.tight_layout()

    return fig


def plot_heatmap(arr, **kwargs):
    """
    Keyword Arguments:
    arr: array to heatmap

    Returns: Figure
    """
    fig = plt.figure(figsize=(6, 6), dpi=72, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)

    sns.heatmap(arr, ax=ax, **kwargs)
    fig.tight_layout()

    return fig


def load_class(full_class_string):
    """
    dynamically load a class from a string
    via https://thomassileo.name/blog/2012/12/21/dynamically-load-python-modules-or-classes/
    """
    class_data = full_class_string.split(".")
    module_path = ".".join(class_data[:-1])
    class_str = class_data[-1]

    module = importlib.import_module(module_path)
    # Finally, we retrieve the Class
    return getattr(module, class_str)


def prepare_batch(batch:Dict, device=None):
    '''
    Input: sample dict
    Output: (input, targets, extras)
    '''
    x, y_true = {}, {}

    for k, v in batch['inputs'].items():
        x[k] = v.to(device)

    for k, v in batch['targets'].items():
        y_true[k] = v.to(device)

    return x, y_true, batch['extra']


def load_model(params, joint_vocab, tables, device):
    model = load_class(params['modelcls'])(joint_vocab, tables, **params).to(device)

    epoch_paths = glob(f'wandb/run-*{params["wandb_id"]}/**/best_checkpoint*.pt', recursive=True)
    latest_epoch_path = sorted(epoch_paths)[-1]
    logging.info('LOAD LATEST BEST MODEL', latest_epoch_path)
    params['model_path'] = latest_epoch_path
    state_dict = torch.load(latest_epoch_path, map_location=device)
    model.load_state_dict(state_dict['model'], strict=False)
    return model


def load_config(wandb_id):
    config_filepath = glob(f'wandb/run-*{wandb_id}/**/config.yaml', recursive=True)
    # Load earliest config to make sure we don't load a modified version.
    file_path = sorted(config_filepath)[0]
    logging.info(f'LOAD EARLIEST CONFIG AT {file_path}')

    with open(file_path) as f:
        c = yaml.load(f)
    config = {k: v['value'] for k, v in c.items() if 'wandb' not in k}

    # Deal with string True, False in wanbd config
    for k, v in config.items():
        if v in ['True', 'False']:
            config[k] = eval(v)

    config['wandb_id'] = wandb_id
    config['config_path'] = file_path
    return config


def load_latest_checkpoint(glob_str, wandb_id):
    model_paths = glob(f'wandb/run-*-{wandb_id}/**/{glob_str}', recursive=True)
    latest_model_path = sorted(model_paths)[-1]
    logging.info(f'LOAD LATEST CHECKPOINT AT {latest_model_path}')

    with open(latest_model_path, 'rb') as checkpoint_file:
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    epoch = latest_model_path.split('_')[2]
    return checkpoint, epoch
