import logging
import re
from typing import Dict, Set

import numpy as np
from torch import nn
from torchtext.vocab import Vectors, Vocab


def feature_string(key):
    '''Normalize label and value text.
    # is used for bin index representation
    = is used for compound representation
    : is used as a seperator
    , unnecessary
    parantheses unnecessary
    '''
    key = re.sub(r'[=\-\s\(\)\[\]\{\}]+', ' ', key)
    key = key.strip().replace(' ', '-')
    return key


def compound_repr(*args):
    return '='.join(args)


def get_counts(vocab_file='embeddings/sentences.mimic3.txt.counts'):
    from collections import Counter
    with open(f'{vocab_file}', 'r') as file:
        counter = Counter()
        for line in file:
            word, count = line.strip().split(' ')
            counter.update({word: int(count)})
    return counter


def get_vocab(emb_path='embeddings/sentences.mimic3.txt.100d.Fasttext.15ws.10neg.vec',
              rand_emb=False,
              min_word_count=10,
              vocab_file='embeddings/sentences.mimic3.txt.counts',
              **kwargs):
    '''returns: torchtext Vocab'''
    if rand_emb:
        logging.info("RANDOM EMBEDDINGS")
        pretrained_embeddings = None
    else:
        pretrained_embeddings = Vectors(emb_path)

    counter = get_counts(vocab_file)
    vocab = Vocab(counter, specials_first=True,
                  vectors=pretrained_embeddings,
                  min_freq=min_word_count,
                  unk_init=nn.init.normal_ if not rand_emb else None,
                  specials=['<pad>', '<unk>'])

    if not rand_emb:
        # initialize unk embedding to normal distribution
        vocab.vectors[vocab.stoi.default_factory()].data.normal_()

    return vocab


def build_vocab(vocab_file='embeddings/sentences.mimic3.txt.counts', min_word_count=10,
                **vocab_kwargs):
    '''returns torchtext Vocab
    '''
    return Vocab(get_counts(vocab_file), specials=['<pad>', '<unk>'], min_freq=min_word_count)


def build_value_vocab(tables, min_word_count=10) -> Dict[str, int]:
    value_vocab: Set[str] = set()
    for table in tables:
        # bins
        value_vocab |= {compound_repr(feature_string(k), str(i+1)) for k, values in table.counts.items() for i, count in enumerate(values) if count > min_word_count}
        # categorical values
        value_vocab |= {compound_repr(feature_string(k), feature_string(value)) for k, values in table.value_counter.items() for value, count in values.items() if (value != 'scalar') and (count > min_word_count)}

    assert len(value_vocab) > 0

    # index = 0 is padding
    return dict(zip(value_vocab, range(1, len(value_vocab))))


class Event:
    def __init__(self, table_source, label, value=None, time=None, include_source=False, **kwargs):
        self.time = time
        self.include_source = include_source
        self.table_source = table_source

        self.label = feature_string(label)

        if value is None:
            self.has_value = False
        else:
            self.has_value = True
            try:
                self.value = float(str(value).strip())
                self.is_scalar = True
            except ValueError:
                self.value = feature_string(value)
                self.is_scalar = False

    @property
    def text(self):
        text = self.label
        if self.include_source:
            text = compound_repr(self.table_source.special_tok, text)
        if self.has_value and not self.is_scalar:
            text = compound_repr(text, feature_string(self.value))
        return text

    def __repr__(self):
        return self.text


class BinnedEvent(Event):
    def __init__(self, table_source, label, value=None, time=None, sep_bin=False, include_source=False, **kwargs):
        self.value_ix = 0
        self.sep_bin = sep_bin
        super().__init__(table_source, label, value, time, include_source)

        try:
            value = float(str(value).strip())
            self.value = value
            # index = 0 is padding
            self.value_ix = np.digitize(value, self.table_source.bins[self.label]) + 1
            self.is_scalar = True
        except KeyError:
            self.is_scalar = True
            pass
        except ValueError:
            self.is_scalar = False

    @property
    def text(self):
        text = super().text
        if self.is_scalar and (self.value_ix > 0) and not self.sep_bin:
            # bins start from 0 in text, i.e. no padding
            text = compound_repr(text, str(self.value_ix - 1))
        return text

    def __repr__(self):
        return self.text


class LabelEvent:
    def __init__(self, table_source, label, value=None, time=None, include_source=False, **kwargs):
        self.time = time
        self.include_source = include_source
        self.table_source = table_source

        self.label = feature_string(label)

    @property
    def text(self):
        text = self.label
        if self.include_source:
            text = compound_repr(self.table_source.special_tok, text)
        self.value_ix = 0
        return text

    def __repr__(self):
        return self.text


class ValuedBinnedEvent(Event):
    def __init__(self, table_source, label, value=None, time=None, sep_bin=False, include_source=False, value_vocab=None, **kwargs):
        self.value_ix = 0
        self.include_source = include_source
        self.sep_bin = sep_bin

        self.label = feature_string(label)

        if value is None:
            self.has_value = False
        else:
            self.has_value = True
            try:
                value = float(str(value).strip())
                self.value = value
                # last value_ix is padding
                value = np.digitize(value, table_source.bins[self.label])
                self.is_scalar = True
            except ValueError:
                value = feature_string(value)
                self.value = value
                self.is_scalar = False
            except KeyError:
                self.is_scalar = True

            self.value_ix = value_vocab.get(compound_repr(self.label, str(value)), 0)

    @property
    def text(self):
        text = super().text
        if self.is_scalar and (self.value_ix > 0) and not self.sep_bin:
            # bins start from 0 in text, i.e. no padding
            text = compound_repr(text, str(self.value_ix - 1))
        return text

    def __repr__(self):
        return self.text


class ValuedEvent(Event):
    def __init__(self, table_source, label, value=None, time=None, sep_bin=False, include_source=False, value_vocab=None, **kwargs):
        self.value_ix = 0

        self.label = feature_string(label)

        if value is None:
            self.has_value = False
        else:
            self.has_value = True
            try:
                value = float(str(value).strip())
                self.value = value
                # last value_ix is padding
                value = np.digitize(value, table_source.bins[self.label])
                self.is_scalar = True
            except ValueError:
                value = feature_string(value)
                self.value = value
                self.is_scalar = False
            except KeyError:
                self.is_scalar = True

            self.value_ix = value_vocab.get(compound_repr(self.label, str(value)), 0)

    @property
    def text(self):
        text = self.label
        return text

    def __repr__(self):
        return self.text
