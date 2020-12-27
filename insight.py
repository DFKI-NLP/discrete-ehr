import csv
import gzip
import os
from collections import defaultdict

import numpy as np
import torch


def get_token_act(output, input, table_key, time_ind):
    t_indices = output['time_inds'][table_key]
    if (t_indices.size(-1) <= np.array(time_ind)).any():
        return
    t_indices = t_indices[:, :, time_ind].cpu()

    # T x C x 1
    m_indices = output['measure_inds'][table_key].expand(-1, -1, t_indices.size(-1)).cpu()
    t_indices = t_indices[:, :m_indices.size(1)]

    padding = (t_indices < 0)
    # 1 x C x 1
    t_indices[t_indices < 0] = 0

    # 1 x C x 1: Get max pool activations for time_ind
    activations = output['activations'][table_key][:, :m_indices.size(1), time_ind].cpu()
    #  1 x C: Flatten
    activations = activations[:, :, 0]
    # Index max pooled measurements at max pooled times
    token_indices = torch.gather(m_indices, 0, t_indices)
    vocab_indices = input[:, t_indices.flatten(), token_indices.flatten(), [0]].detach().cpu()
    # Set time padding to 0
    vocab_indices[padding[:,:,0]] = 0

    tok_act = torch.stack([vocab_indices.float(), activations.float()], -1)
    tok_act[:, padding.flatten(), 0] = 0
    return tok_act


class Insight:
    def __init__(self, vocab, table_keys):
        self.vocab = vocab
        self.table_keys = table_keys
        self.reset()

    def reset(self):
        self.dim_token_activations = defaultdict(list)

    def add_insight(self, output, x, extra):
        # TODO get last timestep pooling as well
        for i, table_key in enumerate(self.table_keys):
            for time_ind in [47, -1]:
                tok_act = get_token_act(output, x[table_key], table_key, [time_ind])
                if tok_act is None:
                    continue

                for dim, (tok, act) in enumerate(tok_act[0]):
                    row = dict(stay=extra['filename'][0],
                               dim=dim,
                               token=self.vocab.itos[int(tok.item())],
                               activation=act.item(),
                               timestep=time_ind)
                    self.dim_token_activations[table_key].append(row)

    def write(self, path):
        os.makedirs(path, exist_ok=True)
        for table_key in self.table_keys:
            with gzip.open(f'{path}/{table_key}_activations.tsv.gz', 'wt') as f:
                fieldnames = ['stay', 'dim', 'token', 'activation', 'timestep']
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
                writer.writeheader()
                for row in self.dim_token_activations[table_key]:
                    writer.writerow(row)
        self.reset()
