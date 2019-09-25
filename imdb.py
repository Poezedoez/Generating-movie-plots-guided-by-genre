import os
import io
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
from collections import defaultdict, Counter, OrderedDict


class OrderedCounter(Counter, OrderedDict):
    """
    Counter that remembers the order elements are first encountered.
    """
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class IMDB(Dataset):
    def __init__(self, data_dir, split, max_sequence_length, min_word_occ):
        super().__init__()

        assert split in ['train', 'val', 'test'], 'Split can only be train/val/test'
    
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.min_word_occ = min_word_occ

        self.raw_data_path = os.path.join(data_dir, 'imdb.{}.csv'.format(split))
        self.data_file = 'imdb.{}.json'.format(split)
        self.vocab_file = 'imdb.vocab.json'

        # Split the movies_genres.csv into separate train, validation, and test
        # data files
        if not os.path.exists(os.path.join(self.data_dir, 'imdb.train.csv')):
            self.split_data()

        if not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print('Preprocessed {} file not found at path {}, creating new...'.format(
                split, os.path.join(self.data_dir, self.data_file)
            ))
            self.create_data()
        else:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)
        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w
    
    def split_data(self):
        """Split the movies_genre.csv data into separate train, validation,
        and test datasets, with an 80-10-10 split.
        """
        print('Splitting IMDB data into separate train, validation, test datasets...')
        raw_data_path = os.path.join(self.data_dir, 'movies_genres.csv')
        df = pd.read_csv(raw_data_path, sep='\t')
        # random state is a seed value
        train = df.sample(frac=0.8, random_state=200)
        # drop the train samples
        rest = df.drop(train.index)
        valid = rest.sample(frac=0.5, random_state=200)
        test = rest.drop(valid.index)
        train.to_csv(os.path.join(self.data_dir, 'imdb.train.csv'))
        valid.to_csv(os.path.join(self.data_dir, 'imdb.val.csv'))
        test.to_csv(os.path.join(self.data_dir, 'imdb.test.csv'))

    def _load_data(self, vocab=True):
        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)
        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def create_data(self):
        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

        tokenizer = TweetTokenizer(preserve_case=False)

        data = defaultdict(dict)
        df = pd.read_csv(self.raw_data_path)
        for _, row in df.iterrows():
            words = tokenizer.tokenize(row['plot'])

            input = ['<sos>'] + words
            input = input[:self.max_sequence_length]

            target = words[:self.max_sequence_length-1]
            target = target + ['<eos>']

            assert len(input) == len(target), "%i, %i"%(len(input), len(target))
            length = len(input)

            input.extend(['<pad>'] * (self.max_sequence_length-length))
            target.extend(['<pad>'] * (self.max_sequence_length-length))

            input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
            target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

            id = len(data)
            data[id]['input'] = input
            data[id]['target'] = target
            data[id]['length'] = length

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):
        assert self.split == 'train', "Vocabulary can only be created for training file."

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        df = pd.read_csv(self.raw_data_path)
        for _, row in df.iterrows():
            words = tokenizer.tokenize(row['plot'])
            w2c.update(words)
        for w, c in w2c.items():
            if c > self.min_word_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocabulary of %i keys created." % len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()