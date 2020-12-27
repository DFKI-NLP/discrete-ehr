from collections import Counter, defaultdict
from pathlib import Path
import os
from random import shuffle
from typing import List

from tqdm.autonotebook import tqdm

from .utils import BinnedEvent, Event
from dataloader.data import MIMICDataset, TabularFeature, get_tables


def extract_sentences(dataset, tables: List[TabularFeature], event_class=Event, suffix='', randomize=False, **params):
    ''' Sentences for each hourly of each table, i.e. seperate sentence for each table.
    '''
    token_counter = Counter()
    Path(f'data/').mkdir(exist_ok=True)
    with open(f'data/sentences{suffix}.{dataset.datasplit}.txt', 'x') as f:
        for sample in tqdm(dataset, desc='extract sentences'):
            for table in tables:
                for t, step in sample['inputs'][table.table]:
                    sentence = []
                    for label, value in step:
                        event = event_class(table, label, value, time=t, **params)
                        sentence.append(event.text)

                    sentence = list(dict.fromkeys(sentence).keys())  # remove duplicate tokens
                    if sentence:
                        token_counter.update(sentence)
                        if randomize:
                            shuffle(sentence)
                        f.write(' '.join(sentence) + '\n')

            # Seperate patients
            f.write('\n')

    with open(f'embeddings/sentences{suffix}.{dataset.datasplit}.counts', 'x') as f:
        for token, count in token_counter.most_common():
            f.write(f'{token} {count}\n')


def extract_merged_sentences(dataset, tables: List[TabularFeature], event_class=Event, suffix='', randomize=False, **params):
    ''' Sentences merged on hourly buckets.
    '''
    token_counter = Counter()
    Path(f'data/').mkdir(parents=True, exist_ok=True)
    with open(f'data/sentences{suffix}.{dataset.datasplit}.txt', 'x') as f:
        for sample in tqdm(dataset, desc='extract sentences'):
            sentences = defaultdict(list)
            for table in tables:
                for t, step in sample['inputs'][table.table]:
                    for label, value in step:
                        event = event_class(table, label, value, time=t, **params)
                        sentences[t].append(event.text)

            for sentence in sentences.values():
                sentence = list(dict.fromkeys(sentence).keys())  # remove duplicate tokens
                if sentence:
                    token_counter.update(sentence)
                    if randomize:
                        shuffle(sentence)
                    f.write(' '.join(sentence) + '\n')

            # Seperate patients
            f.write('\n')

    with open(f'embeddings/sentences{suffix}.{dataset.datasplit}.counts', 'x') as f:
        for token, count in token_counter.most_common():
            f.write(f'{token} {count}\n')


def extract_patient_sentences(dataset, tables: List[TabularFeature], event_class=Event, suffix='', randomize=False, **params):
    ''' Sentences merged on hourly buckets.
    '''
    token_counter = Counter()
    Path(f'data/').mkdir(parents=True, exist_ok=True)
    with open(f'data/sentences{suffix}.{dataset.datasplit}.txt', 'x') as f:
        for sample in tqdm(dataset, desc='extract sentences'):
            steps = []
            for table in tables:
                for t, step in sample['inputs'][table.table]:
                    steps.append((t, table, step))

            steps = sorted(steps, key=lambda x: x[0])

            sentence = []
            for t, table, step in steps:
                for label, value in step:
                    event = event_class(table, label, value, time=t, **params)
                    sentence.append(event.text)

            if sentence:
                token_counter.update(sentence)
                if randomize:
                    shuffle(sentence)
                f.write(' '.join(sentence) + '\n')

    with open(f'embeddings/sentences{suffix}.{dataset.datasplit}.counts', 'x') as f:
        for token, count in token_counter.most_common():
            f.write(f'{token} {count}\n')


def extract_vocab(dataset, tables, event_class=Event, suffix='', **params):
    token_counter = Counter()
    for sample in tqdm(dataset, desc='extract vocab'):
        for table in tables:
            for t, step in sample['inputs'][table.table]:
                sentence = set()
                for label, value in step:
                    event = event_class(table, label, value, time=t, **params)
                    sentence.add(event.text)
                token_counter.update(sentence)

    os.makedirs('embeddings', exist_ok=True)
    with open(f'embeddings/sentences{suffix}.{dataset.datasplit}.txt.counts', 'x') as f:
        for token, count in token_counter.most_common():
            f.write(f'{token} {count}\n')

    return token_counter
