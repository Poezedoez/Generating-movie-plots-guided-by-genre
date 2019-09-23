import pandas as pd
import numpy as np
import torch
import json
import nltk
from collections import defaultdict

END_OF_SENTENCE_TAG = '<EOS>'
UNKNOWN_TAG = '<UNK>'
END_OF_PARAGRAPH_TAG = '<EOP>'
VOCAB_LOCATION = 'assets/vocab.json'
PADDING_TAG = 0

def map_word(word, index, sample_length):
  word = word.lower()
  if word == ".":
    return END_OF_SENTENCE_TAG

  if (index - 1) == sample_length:
    return END_OF_PARAGRAPH_TAG

  return word

def create_vocab():
  # if word is unknown, return corresponding index of <UNK>
  vocab = defaultdict(lambda: 1)

  vocab[PADDING_TAG] = 0
  vocab[UNKNOWN_TAG] = 1
  vocab[END_OF_SENTENCE_TAG] = 2
  vocab[END_OF_PARAGRAPH_TAG] = 3

  for _, row in df.iterrows():
    tokenized = nltk.word_tokenize(row['plot'])
    for word, index in zip(tokenized, range(len(tokenized))):
      transformed_word = map_word(word, index, len(tokenized))
      if (transformed_word not in vocab):
        vocab[transformed_word] = len(vocab)

  return vocab

def load_vocab():
  try:
    vocab = json.load(open(VOCAB_LOCATION))
  except FileNotFoundError:
    vocab = create_vocab()
    with open(VOCAB_LOCATION, 'w') as f:
      json.dump(vocab, f)
  return vocab

class DataHandler:

  def __init__(self, filename, train_val_test_split, batch_size):
    df = pd.read_csv(filename, sep='\t')
    self.vocab = load_vocab()
    data_size = len(df)
    indices = np.random.shuffle(list(range(data_size)))
    assert sum(train_val_test_split) == 1, "Sum of split should be one."
    train_size = int(train_val_test_split[0] * data_size)
    val_size = int(train_val_test_split[1] * data_size)
    test_size = int(train_val_test_split[2] * data_size)

    self.batch_size = batch_size
    self.batches_in_epoch = int(train_size / batch_size)
    self.train_data = df[:train_size]
    self.validation_data = df[train_size:(train_size+val_size)]
    self.test_data = df[(train_size+val_size+test_size):]

  def create_feature_vector(self, sample):
    tokenized = nltk.word_tokenize(sample)
    return torch.LongTensor([
      self.vocab[map_word(word, index, len(tokenized))]
      for word, index in zip(tokenized, range(len(tokenized)))
    ])

  def get_data_length(self, data_type):
    assert data_type in ["train", "test", "val"], "Incorrect data type"

    if (data_type == "train"):
      return len(self.train_data)

    if (data_type == "val"):
      return len(self.val_data)

    if (data_type == "test"):
      return len(self.test_data)

  def load_batch(self, data_type="train"):
    assert data_type in ["train", "test", "val"], "Incorrect data type"

    if (data_type == "train"):
      sampled = self.train_data.sample(self.batch_size)

    if (data_type == "val"):
      sampled = self.val_data.sample(self.batch_size)

    if (data_type == "test"):
      sampled = self.test_data.sample(self.batch_size)

    features = [
      self.create_feature_vector(sample['plot'])
      for _, sample in sampled.iterrows()
    ]

    return torch.nn.utils.rnn.pad_sequence(features, batch_first=True)

