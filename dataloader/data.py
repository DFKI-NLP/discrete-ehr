import math
import torch
import os
import pickle
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm.autonotebook import tqdm
from sklearn.preprocessing import KBinsDiscretizer

from dataloader.utils import BinnedEvent, Event
from dataloader.labels import Label


class MIMICDataset(data.Dataset):
    def __init__(self,
                 base_path='mimic3-benchmarks/data/multitask',
                 datasplit='train', mode='TRAIN',
                 datalist_file='train_listfile.csv',
                 tables: List=[], labels: Dict[str, Label]={},
                 limit=None, numericalize=False, use_cache=True,
                 **kwargs):
        '''
        Dataset for in hospital mortality task
        Each sample is a patient's single visit.
        '''
        self.mode = mode
        self.base_path = Path(base_path)
        self.datasplit = datasplit
        self.use_cache = use_cache

        self.file_df = pd.read_csv(self.base_path / datalist_file, dtype=dict(value='object'))
        self.file_df['subj'] = self.file_df.filename.apply(lambda x: int(x.split("_")[0]))
        self.file_df['episode'] = self.file_df.filename.apply(lambda x: x.split("_")[1])
        # use only benchmark labels
        # self.file_df = self.file_df[self.file_df.filename.str.count('_') == 2]
        self.file_df = self.file_df.set_index(['subj', 'episode'])
        if limit is not None:
            if isinstance(limit, int):
                self.file_df = self.file_df.iloc[:limit]
            elif isinstance(limit, list):
                self.file_df = self.file_df[self.file_df.filename.isin(limit)]
            elif isinstance(limit, str):
                self.file_df = self.file_df[self.file_df.filename.str.match(limit)]

        self.file_df['idx'] = range(len(self.file_df))

        filepath = self.base_path / self.datasplit / 'demogfile.csv'
        self.demog_df = pd.read_csv(filepath).set_index('filename')

        self.tables: List[TabularFeature] = tables

        self.labels = labels
        for label in labels.values():
            self.file_df[label.task] = label.preprocess(self.file_df)

        self.rng = np.random.default_rng(42)
        self.numericalize = numericalize
        self.examples: List[Dict] = []
        self.datalist_filename = datalist_file.replace('.csv', '').replace('/', '_')

    def load_patient(self, subj, episode):
        with open(f"cache/{self.datasplit}/{subj}_{episode}.pl", 'rb') as f:
            return pickle.load(f)

    def load_in_memory(self):
        self.examples = [self.get_example(i) for i in tqdm(range(len(self)))]

    def save_patient(self, subj, episode, patient):
        os.makedirs(f"cache/{self.datasplit}", exist_ok=True)
        with open(f"cache/{self.datasplit}/{subj}_{episode}.pl", 'xb') as f:
            return pickle.dump(patient, f)

    def get_by_filename(self, filename):
        return self.get_example(self.file_df.reset_index()[self.file_df.reset_index().filename == filename].index[0])

    def get_example(self, index):
        '''
        Input:
            index: int
        Output:
            a dictionary for an example with {'inputs': {<table_key>: Tensor or List},
                                              'targets': {<label_key>: Tensor or List},
                                              'extra': {key: List}}
        '''
        if self.use_cache:
            try:
                df = self.file_df.iloc[index]
                subj, episode = df.name
                example = self.load_patient(subj, episode)
            except FileNotFoundError:
                example = self.make_example(index)
        else:
            example = self.make_example(index)

        if self.numericalize:
            numericalized_input = {}
            for table in self.tables:
                numericalized_input[table.table] = table.numericalize(example['inputs'], self.mode)
            example['inputs'] = numericalized_input

            numericalized_targets = {}
            for label_key, label in self.labels.items():
                numericalized_targets[label_key] = label.numericalize(example['targets'].get(label_key, None))
            example['targets'] = numericalized_targets

        return example

    def make_example(self, index):
        df = self.file_df.iloc[index]
        subj, episode = df.name
        pat = df.to_dict()

        patient = {}
        patient['extra'] = {}
        patient['extra']['los'] = pat['length of stay']
        patient['extra']['subj'] = subj
        patient['extra']['episode'] = episode
        patient['extra']['idx'] = index
        patient['extra']['filename'] = pat['filename']

        # get label
        patient['targets'] = {}
        for task, label in self.labels.items():
            patient['targets'][f'{task}'] = pat[task]

        demog = self.demog_df.loc[pat['filename']]
        patient['extra'].update(dict(demog[['Ethnicity', 'Gender', 'Age', 'Height', 'Weight']]))

        patient['inputs'] = {}
        length = math.ceil(patient['extra']['los'])
        for table in self.tables:
            try:
                filepath = self.base_path / self.datasplit
                X = table.extract(filepath, pat['filename'], length=length)
            except (FileNotFoundError, IndexError):
                X = table.get_empty_pad(length)
            patient['inputs'][table.table] = X

        if self.use_cache:
            self.save_patient(subj, episode, patient)
        return patient

    def split(self, ratio=0.85):
        df = self.file_df.reset_index()
        subjects = df.subj.unique()
        left_subjects = self.rng.choice(subjects, size=int(len(subjects) * ratio), replace=False)
        left_mask = df.subj.isin(left_subjects)
        left_indices = df[left_mask].index
        right_indices = df[~left_mask].index
        left = data.Subset(self, left_indices)
        left.file_df = df[left_mask]
        left.demog_df = left.dataset.demog_df
        left.datalist_filename = left.dataset.datalist_filename + 'LEFT'

        right = data.Subset(self, right_indices)
        right.file_df = df[~left_mask]
        right.demog_df = right.dataset.demog_df
        right.datalist_filename = right.dataset.datalist_filename + 'RIGHT'

        return left, right

    def get_pat(self, subj, episode):
        '''
        Input:
            subj: id as int
            episode: episode as str, e.g. episode1
        Output:
            sample
        '''
        return self[self.file_df.loc[(subj, episode)].idx]

    def __len__(self):
        return len(self.file_df)

    def __getitem__(self, index):
        if self.examples:
            example = self.examples[index]
        else:
            example = self.get_example(index)

        return example


class DemographicsFeature:
    def __init__(self, table, suffix='.mimic3', normalized_indices=[0, 2, 4],
                 **kwargs):
        self.rng = np.random.RandomState(5)

        self.n_dims = 8
        self.input_dims = 8

        self.suffix = suffix
        self.table = table
        self.normalized_indices = normalized_indices

    def load(self, **kwargs):
        with open(f'cache/{self.table}{self.suffix}.params', 'rb') as f:
            params = pickle.load(f)
            self.means = params['means']
            self.stds = params['stds']

    def fit(self, dataset, **kwargs):
        values = []
        for sample in tqdm(dataset, desc='extract values'):
            sample = sample['inputs'][self.table]
            values.append(sample[self.normalized_indices])
        values = np.array(values)

        min_q, max_q = np.quantile(values, [0.01, 0.99])
        if min_q < max_q:
            values = values[(min_q < values) & (values < max_q)]

        self.means = values.mean(0)
        self.stds = values.std(0)

    def save(self):
        os.makedirs('cache', exist_ok=True)
        with open(f'cache/{self.table}{self.suffix}.params', 'xb') as f:
            pickle.dump(dict(means=self.means, stds=self.stds), f)

    def get_empty_pad(self, *args, **kwargs):
        return np.zeros(self.n_dims, dtype=np.float32)

    def extract(self, filepath, filename, **kwargs):
        '''
        returns [weight, weight exists, height, height exists, age, age exists, gender, gender_exists]
        where gender is F: 0, M: 1
        '''
        filepath = filepath / 'demogfile.csv'
        df = pd.read_csv(filepath).set_index('filename')

        row = df.loc[filename][['Weight', 'Height', 'Age', 'Gender']]
        exist_flag = ~pd.isna(row[:4])
        row = row.fillna(0)
        if row[3] not in [1, 2]:
            exist_flag[3] = 0
        row[3] = max(row[3] - 1, 0)

        dem = np.array(sum([list(z) for z in zip(row, exist_flag)], []), dtype=np.float32)
        return dem

    def numericalize(self, arr, mode):
        '''returns [weight, weight exists, height, height exists, age, age exists, gender, gender exists]'''
        arr = arr.get(self.table, [(0, [])])
        arr[self.normalized_indices] = (arr[self.normalized_indices] - self.means) / self.stds
        return torch.from_numpy(arr)


class TabularFeature:
    def __init__(self, table, label, value,
                 event_dropout=0., step_dropout=0.,
                 sep_bin=False, suffix='.mimic3', special_tok='',
                 event_class=Event,
                 include_source=False,
                 **kwargs):
        self.rng = np.random.default_rng(5)

        self.event_dropout = event_dropout
        self.step_dropout = step_dropout

        self.n_dims = 2
        self.dtype = torch.long
        self.suffix = suffix

        self.vocab = None
        self.embedder = None
        self.sep_bin = sep_bin

        self.values = defaultdict(list)
        self.value_counter = defaultdict(Counter)

        self.special_tok = special_tok
        self.table = table
        self.label = label
        self.value = value
        self.fields = [label] if value is None else [label, value]
        self.event_class = event_class
        self.include_source = include_source
        self.value_vocab = None

    def get_empty_pad(self, length):
        '''numericalize handles padding. Returns empty sequence.'''
        return [(t, []) for t in range(math.ceil(length))]

    def _timestep_transform(self, df, length):
        df['timestep'] = pd.cut(df['Hours'], range(length+1))
        df['timestep'] = df['timestep'].apply(lambda x: x.left)
        return df

    def extract(self, filepath, filename, length):
        filepath = filepath / filename.replace('.csv', f'_{self.table}.csv')
        filepath = glob(str(filepath))[0]
        df = pd.read_csv(filepath)

        df = df.dropna(subset=self.fields)
        df = df[self.fields + ['Hours']]

        df = self._timestep_transform(df, length)

        # Drop exact same lines in same timestep
        df = df.drop_duplicates(['timestep'] + self.fields, keep='last')

        df = df.groupby('timestep', as_index=True).apply(lambda x: list(zip(*[x[c] for c in self.fields])))
        return list(df.iteritems())

    def extract_values(self, dataset: MIMICDataset, event_class=Event):
        for sample in tqdm(dataset, desc='extract values'):
            for t, step in sample['inputs'][self.table]:
                for label, value in step:
                    event = event_class(self, label, value, time=t)
                    if event.is_scalar:
                        self.value_counter[event.label].update(['scalar'])
                    else:
                        self.value_counter[event.label].update([event.value])
                    self.values[event.label].append(event.value)

    def fit(self, dataset: MIMICDataset, **kwargs):
        self.extract_values(dataset)

    def numericalize(self, arr, mode, pad=True):
        '''arr is a list of buckets with events
        returns X:
        X: Tensor of shape L, L, C if pad=True
        else L length list of tensors of shape L, C
        '''
        arr = arr.get(self.table, [(0, [])])
        X = []
        for t, visit in arr:
            x = []
            for i, (event_label, value) in enumerate(visit):
                # input dropout for steps
                if (i == 0) and (mode == 'TRAIN') and (self.rng.random() < self.step_dropout):
                    break
                # input dropout for events
                if mode == 'TRAIN' and (self.rng.random() < self.event_dropout):
                    continue

                x_event = self.numericalize_event(t, event_label, value)
                if x_event is None:
                    continue

                x.append(x_event)

            if len(x) == 0:
                x.append(torch.zeros(self.n_dims, dtype=self.dtype))

            X.append(torch.stack(x, 0))

        if len(X) == 0:
            X.append(torch.zeros(self.n_dims, dtype=self.dtype))

        if pad:
            X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
            assert self.dtype == X.dtype, self.table

        return X

    def numericalize_event(self, t, event_label, value):
        event = self.event_class(self, event_label, value, time=t, sep_bin=self.sep_bin, include_source=self.include_source, value_vocab=self.value_vocab)
        # Unknown tokens default to 1
        token_ix = self.vocab.stoi[event.text]
        # Skip unknown tokens
        if token_ix == 1:
            return None

        return torch.tensor([token_ix, event.value_ix], dtype=torch.long)

    def load(self):
        with open(f'cache/med_value_counts.{self.table}{self.suffix}.txt') as f:
            f.readline()  # skip header
            for line in f:
                data = line.strip().split(' ')
                token = data[0]
                self.value_counter[token].update(dict(zip(data[1].split(':'), map(int, data[2].split(':')))))

    def save(self):
        os.makedirs('cache', exist_ok=True)
        with open(f'cache/med_value_counts.{self.table}{self.suffix}.txt', 'x') as f:
            f.write('TOKEN values counts\n')
            for label, counter in self.value_counter.items():
                values_string = ":".join(counter.keys())
                counts_string = ":".join([str(c) for c in counter.values()])

                f.write(f'{label} {values_string} {counts_string}\n')


class TextTabularFeature(TabularFeature):
    def __init__(self, *args, lm='emilyalsentzer/Bio_ClinicalBERT', **kwargs):
        super().__init__(*args, **kwargs)

        self.n_dims = 768
        self.dtype = torch.float

    def load(self):
        pass

    def save(self):
        pass

    def extract(self, filepath, filename, length):
        filepath = filepath / filename.replace('.csv', f'_{self.table}.csv')
        filepath = glob(str(filepath))[0]
        df = pd.read_csv(filepath)
        embeddings = np.load(filepath.replace('.csv', '_precomputed_ho.npy'))

        df['ENCODING'] = list(embeddings)

        df = df.dropna(subset=self.fields)
        df = df[self.fields + ['Hours']]

        df = self._timestep_transform(df, length)

        df = df.groupby('timestep', as_index=True).apply(lambda x: list(zip(*[x[c] for c in self.fields])))
        return list(df.iteritems())

    def numericalize_event(self, t, event_label, value):
        return torch.tensor(value)


class BinnedTabularFeature(TabularFeature):
    def __init__(self, *args, suffix='.mimic3', n_bins=10, strategy='uniform', **kwargs):
        '''
        n_bins: number of bins to discritize continuous values into, includes outlier bins.
        '''
        super().__init__(*args, **kwargs)
        self.suffix = suffix
        self.n_bins = n_bins
        self.strategy = strategy
        self.counts = {}
        self.bins = {}

    def fit(self, dataset: MIMICDataset, **kwargs):
        super().fit(dataset, **kwargs)
        self.extract_bins(**kwargs)

    def extract_bins(self, outlier_quantile=[0.01, 0.99], do_plot=False):
        '''
        do_plot: flag to plot histogram of values on bins
        '''
        for key, values in tqdm(self.values.items(), desc='extract bins'):
            scalar_values = []
            for value in values:
                try:
                    scalar_values.append(float(value))
                except ValueError:
                    continue

            if not scalar_values or (len(scalar_values) < 10):
                continue

            scalar_values = np.array(scalar_values)[:, None]

            min_q, max_q = np.quantile(scalar_values, outlier_quantile)
            if min_q < max_q:
                scalar_values = np.clip(scalar_values, min_q, max_q)

            d = KBinsDiscretizer(n_bins=min(len(scalar_values), self.n_bins-2),
                                 encode='ordinal', strategy=self.strategy)
            x = d.fit_transform(scalar_values)
            self.bins[key] = d.bin_edges_[0]
            self.counts[key] = np.bincount(x[:, 0].astype(int), minlength=d.n_bins_[0])

            if do_plot: #and np.random.random() < 0.01:
                self.plot_bin(key)

    def plot_bin(self, key, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.subplot(111)
        _bins = self.bins[key][:-1]
        ax.bar(_bins, self.counts[key], np.diff(self.bins[key]), align='edge')
        ax.set_xticks(self.bins[key])
        ax.set_xticklabels(np.round(self.bins[key], 2))
        ax.set_title(f'{self.table} {key} ({min(_bins):.1f}-{max(_bins):.1f} N={self.counts[key]})')
        return ax

    def save(self):
        super().save()
        with open(f'cache/med_bin_edges.{self.table}{self.suffix}.{self.strategy}.{self.n_bins}bins.txt', 'x') as f:
            f.write('TOKEN edges counts\n')
            for token, bin_edges in self.bins.items():
                edges_string = ":".join([str(e) for e in bin_edges])
                counts_string = ":".join([str(c) for c in self.counts[token]])

                f.write(f'{token} {edges_string} {counts_string}\n')

    def load(self):
        super().load()
        with open(f'cache/med_bin_edges.{self.table}{self.suffix}.{self.strategy}.{self.n_bins}bins.txt') as f:
            # skip header
            f.readline()
            for line in f:
                data = line.strip().split(' ')
                token = data[0]
                self.bins[token] = [float(e) for e in data[1].split(':')]
                self.counts[token] = [float(e) for e in map(float, data[2].split(':'))]


class NormalizedMultivariateFeature(BinnedTabularFeature):
    '''Multi-variate input'''
    def numericalize(self, arr, mode, pad=True):
        '''arr is a list of buckets with events
        returns X:
        X: Tensor of shape L, C
        '''
        arr = arr.get(self.table, [(0, [])])
        X = []
        for t, visit in arr:
            x = torch.zeros(2 * len(self.vocab))  # vector includes presence indicators
            for i, (event_label, value) in enumerate(visit):
                # input dropout for steps
                if (i == 0) and (mode == 'TRAIN') and (self.rng.random() < self.step_dropout):
                    break

                try:
                    event = self.event_class(self, event_label, value, time=t, sep_bin=self.sep_bin, include_source=self.include_source)
                    token_ix = self.vocab.stoi[event.text]

                    # skip unknown tokens
                    if token_ix == 1:
                        continue

                    if event.is_scalar:
                        # normalize
                        x[token_ix] = event.value / np.median(self.bins[event.label])
                        # add indicator feature using the bin, skip 0 bin index
                        x[token_ix * 2] = event.value_ix + 1 / self.n_bins
                    elif event.has_value:
                        # one-hot encoding for categorical values
                        x[token_ix] = 1
                        x[token_ix * 2] = 1

                except KeyError as e:
                    print(event_label, value, 'keyerror', e)
                    continue

            X.append(x, dtype=torch.float)

        if len(X) == 0:
            X.append(torch.zeros(2 * len(self.vocab), dtype=torch.float))

        return X


class JointTabularFeature:
    def __init__(self, tables: List[TabularFeature]):
        self.tables = tables
        self.table = '_'.join([table.table for table in tables])

    def numericalize(self, arr, mode, pad=True):
        X_tables = []
        for table in self.tables:
            X_tables.append(table.numericalize(arr, mode, pad=False))

        X = []
        for timesteps in zip(*X_tables):
            X.append(torch.cat(timesteps, 0))

        if pad:
            X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        return X

    def extract(self, *args, **kwargs):
        raise NotImplementedError


def get_tables(input_tables=['dem', 'CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS'],
               vocab=None, load=False, joint_tables=False, suffix='',
               **kwargs):
    '''
    Returns a fixed order tables filtered with the input_tables parameter: dem, chart, lab, output, prescr, inpute
    '''
    chart = BinnedTabularFeature('CHARTEVENTS', 'LABEL', 'VALUE', special_tok='C', suffix=suffix, **kwargs)
    chart.vocab = vocab
    lab = BinnedTabularFeature('LABEVENTS', 'LABEL', 'VALUE', special_tok='L', suffix=suffix, **kwargs)
    lab.vocab = vocab
    output = BinnedTabularFeature('OUTPUTEVENTS', 'LABEL', 'VALUE', special_tok='O', suffix=suffix, **kwargs)
    output.vocab = vocab
    prescr = BinnedTabularFeature('PRESCRIPTIONS', 'DRUG', 'DOSE_VAL_RX', special_tok='P', suffix=suffix, **kwargs)
    prescr.vocab = vocab
    inpute = BinnedTabularFeature('INPUTEVENTS_*', 'LABEL', 'AMOUNT', special_tok='I', suffix=suffix, **kwargs)
    inpute.vocab = vocab
    note = TextTabularFeature('NOTEEVENTS', 'CATEGORY', 'ENCODING', special_tok='N', lm='emilyalsentzer/Bio_ClinicalBERT', **kwargs)
    dem = DemographicsFeature('dem', normalized_indices=[0, 2, 4], suffix=suffix, **kwargs)

    # This order is kept and dependent upon through out the code!
    all_tables = [dem, chart, lab, output, prescr, inpute, note]
    tables = [table for table in all_tables if table.table in input_tables]

    if load:
        for table in tables:
            table.load()

    if joint_tables:
        tabular_features = [table for table in tables if isinstance(table, BinnedTabularFeature)]
        joint_tables = [JointTabularFeature(tabular_features)]
        if dem in tables:
            joint_tables = [dem] + joint_tables
        tables = joint_tables

    return tables


if __name__ == '__main__':
    DEVICE = 'cpu'
    SUFFIX = '.mimic3'
    N_BINS = 9
    VOCAB_FILE = 'embeddings/sentences.mimic3.counts'

    from dataloader.labels import get_labels
    from dataloader.utils import build_vocab
    joint_vocab = build_vocab(VOCAB_FILE, min_word_count=10)
    tables = get_tables(['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS', 'dem'], vocab=joint_vocab, load=True,
                          suffix=SUFFIX, n_bins=N_BINS, sep_bin=False, event_class=BinnedEvent)
    labels = get_labels(DEVICE)

    train_set = MIMICDataset(tables=tables, limit='^(111)', n_bins=N_BINS, labels=labels, numericalize=False)

    print(len(train_set))

    sample = train_set.get_by_filename('11123_episode1_timeseries.csv')
    for sample in train_set:
        sample
        break
