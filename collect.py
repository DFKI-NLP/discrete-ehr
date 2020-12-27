import logging
import argparse
from glob import glob
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from insight import Insight
from dataloader.data import MIMICDataset, get_tables, TabularFeature, DemographicsFeature
from dataloader.labels import Label, get_labels
from dataloader.utils import BinnedEvent, get_vocab


def parse_arguments():
    parser = argparse.ArgumentParser(description='collect predictions and insights for a trained model')
    parser.add_argument('wandb_ids', nargs='+', default=None, help='experiment id to get the latest best model and config')
    parser.add_argument('--dev', default=False, action='store_true')
    return parser.parse_args()


def collect(wandb_id, model: torch.nn.Module, device,
            train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
            tables: List[TabularFeature], labels: Dict[str, Label],
            tboardwriter=None,
            **params):
    model_path = Path(glob(f'wandb/*{wandb_id}/**/best*.pt')[-1])
    model_epoch = model_path.name.split('_')[2]
    save_dir = str(Path(model_path).parent)

    insight = Insight(model.timestep_encoder.event_encoder.vocab, [table.table for table in tables if not isinstance(table, DemographicsFeature)])

    def store_predictions(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y_true, extra = utils.prepare_batch(batch, device)

            preds, output = model(*x.values())
            output.update({"y_pred": preds,
                           "y_true": y_true})
            output.update(extra)

            for label in labels.values():
                label.add_result(output)
                extra.update(label.dict_repr(output))

            engine.state.embeddings.append(output['patient'][:, -1].cpu())
            sorting = defaultdict(lambda: 10)
            sorting['filename'] = 0
            extra = dict(sorted(extra.items(), key=lambda x: sorting[x[0]]))
            engine.state.metadata.extend(list(zip(*extra.values())))
            engine.state.metadata_header = extra.keys()

            insight.add_insight(output, x, extra)

            return output

    prediction_collector = Engine(store_predictions)
    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(prediction_collector)

    # PERSISTENCE
    PERSISTANCE_DICT = {'model': model}

    def load_checkpoint(prefix):
        checkpoint, _ = utils.load_latest_checkpoint(prefix, wandb_id)
        Checkpoint.load_objects(to_load=PERSISTANCE_DICT, checkpoint=checkpoint,
                                strict=True)

    load_checkpoint('best*')

    @prediction_collector.on(Events.STARTED)
    def init_embeddings_collection(engine):
        prediction_collector.state.embeddings = []
        prediction_collector.state.metadata = []
        prediction_collector.state.metadata_header = None

    @prediction_collector.on(Events.COMPLETED)
    def write_predictions(engine):
        DATA_SOURCE = engine.state.dataloader.dataset.datalist_filename
        for label in labels.values():
            label.save_results(f'{save_dir}/{DATA_SOURCE}_predictions', model_epoch)

        metrics = engine.state.metrics
        scalar_metrics = [(k, v) for k, v in metrics.items() if 'plot' not in k]
        logging.info(f"{DATA_SOURCE} Results - Epoch: {model_epoch},  {scalar_metrics}")

        tboardwriter.add_embedding(torch.cat(engine.state.embeddings, 0),
                                   engine.state.metadata,
                                   metadata_header=engine.state.metadata_header,
                                   tag=f'patient_{DATA_SOURCE}',
                                   global_step=0)
        engine.state.embeddings, engine.state.metadata, engine.state.metadata_header = [], [], None

        insight.write(save_dir+f'/insight_{model_epoch}/')

    prediction_collector.run(val_loader)
    prediction_collector.run(test_loader)

    return {}


def run(wandb_id, data_path, dev, **params):
    DEVICE = 'cuda' if not dev else 'cpu'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_level = 'DEBUG' if dev else 'INFO'
    logging.info("---NEW TRAINING---")

    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    logging.info(f'TRAIN: {params}')

    labels = get_labels(DEVICE, **params)

    joint_vocab = get_vocab(**params)
    tables = get_tables(vocab=joint_vocab,
                        load=True,
                        event_class=BinnedEvent,
                        **params)

    if not params['random_split']:
        train_set = MIMICDataset(data_path, 'train', datalist_file='train_listfile.csv', mode='TRAIN',
                                 tables=tables, labels=labels,
                                 limit='^(111)' if dev else None,
                                 numericalize=True,
                                 )
        val_set = MIMICDataset(data_path, 'train', datalist_file='val_listfile.csv', mode='EVAL',
                               tables=tables, labels=labels,
                               limit='^(11).*' if dev else None,
                               numericalize=True,
                               )
    else:
        logging.warning('DATA: RANDOM SPLIT')
        train_set = MIMICDataset(data_path, 'train', datalist_file='train/listfile.csv', mode='TRAIN',
                                 tables=tables, labels=labels,
                                 limit='^(11)' if dev else None,
                                 numericalize=True,
                                 )
        train_set, val_set = train_set.split()

    test_set = MIMICDataset(data_path, 'test', datalist_file='test_listfile.csv', mode='EVAL',
                            tables=tables, labels=labels,
                            limit='^(11|22)' if dev else None,
                            numericalize=True,
                            )

    from functools import partial
    logging.info(f'Training on {len(train_set)}   Validating on {len(val_set)}')

    sampler_class = utils.load_class(params['sampler'])
    sampler = sampler_class(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=params['batch_size'],
                                               collate_fn=partial(utils.min_batch,
                                                                  tables=tables,
                                                                  labels=labels,
                                                                  limit=720),
                                               sampler=sampler,
                                               num_workers=params['num_workers'] if not dev else 0, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1,
                                             collate_fn=partial(utils.pad_batch,
                                                                tables=tables,
                                                                labels=labels,
                                                                limit=None),
                                             shuffle=False, num_workers=params['num_workers'] if not dev else 0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              collate_fn=partial(utils.pad_batch,
                                                                 tables=tables,
                                                                 labels=labels,
                                                                 limit=None),
                                              shuffle=False, num_workers=params['num_workers'] if not dev else 0, pin_memory=True)

    model, device = utils.create_model_on_gpu(params, DEVICE, tables=tables,
                                              joint_vocab=joint_vocab)
    logging.info(device)

    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    tboardwriter = SummaryWriter(f'runs/collect_{wandb_id}')

    collect(wandb_id, model, device, train_loader, val_loader, test_loader,
            tables, labels, tboardwriter, **params)

args = parse_arguments()

for wandb_id in args.wandb_ids:
    print(wandb_id)
    config = utils.load_config(wandb_id)
    config.update(vars(args))

    print(config)
    run(**config)
