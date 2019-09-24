import pandas as pd
import numpy as np
import torch
import json
import nltk
import sys
from collections import defaultdict
from collections import Counter
from nltk.tokenize import sent_tokenize

END_OF_SENTENCE_TAG = '<EOS>'
UNKNOWN_TAG = '<UNK>'
END_OF_PARAGRAPH_TAG = '<EOP>'
VOCAB_LOCATION = 'assets/vocab.json'
PADDING_TAG = '<PAD>'

def map_word(word, index, sample_length):
  word = word.lower()
  if word == ".":
    return END_OF_SENTENCE_TAG

  if (index - 1) == sample_length:
    return END_OF_PARAGRAPH_TAG

  return word

def create_vocab(df):
  # if word is unknown, return corresponding index of <UNK>
  vocab = defaultdict(lambda: 1)
  counter = Counter()

  vocab[PADDING_TAG] = 0
  vocab[UNKNOWN_TAG] = 1
  vocab[END_OF_SENTENCE_TAG] = 2
  vocab[END_OF_PARAGRAPH_TAG] = 3

  for _, row in df.iterrows():
    tokenized = nltk.word_tokenize(row['plot'])
    for word, index in zip(tokenized, range(len(tokenized))):
      transformed_word = map_word(word, index, len(tokenized))
      counter[transformed_word] += 1
      if (transformed_word not in vocab) and counter[transformed_word] > 50:
        vocab[transformed_word] = len(vocab)

  return vocab

def load_vocab(df):
  try:
    vocab = json.load(open(VOCAB_LOCATION))
  except FileNotFoundError:
    vocab = create_vocab(df)
    with open(VOCAB_LOCATION, 'w') as f:
      json.dump(vocab, f)
  return vocab

class DataHandler:

  def __init__(self, filename, train_val_test_split, batch_size, max_sequence_length):
    df = pd.read_csv(filename, sep='\t')
    self.vocab = load_vocab(df)
    self.max_sequence_length = max_sequence_length

    data = self.plot_to_sentences(df)

    data_size = len(data)
    indices = np.random.shuffle(list(range(data_size)))
    assert sum(train_val_test_split) == 1, "Sum of split should be one."
    train_size = int(train_val_test_split[0] * data_size)
    val_size = int(train_val_test_split[1] * data_size)
    test_size = int(train_val_test_split[2] * data_size)

    self.batch_size = batch_size
    self.batches_in_epoch = int(train_size / batch_size)
    self.train_data = data[:train_size]
    self.val_data = data[train_size:(train_size+val_size)]
    self.test_data = data[(train_size+val_size+test_size):]
    

  def plot_to_sentences(self, df):
    return [
      sentence
      for _, movie in df.iterrows()
      for sentence in sent_tokenize(movie['plot'])
      if len(sent_tokenize(sentence)) < self.max_sequence_length
    ]

  def create_feature_vector(self, sample):
    tokenized = nltk.word_tokenize(sample)
    return torch.LongTensor([
      # please fix later
      self.vocab.get(map_word(word, index, len(tokenized)), 1)
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
      sampled = np.random.choice(self.train_data, self.batch_size)

    if (data_type == "val"):
      sampled = np.random.choice(self.val_data, self.batch_size)

    if (data_type == "test"):
      sampled = np.random.choice(self.test_data, self.batch_size)

    features = [
      self.create_feature_vector(sample)
      for sample in sampled
    ]

    return torch.nn.utils.rnn.pad_sequence(features, batch_first=True)

