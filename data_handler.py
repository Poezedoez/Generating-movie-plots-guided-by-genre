import pandas as pd
import torch
import json
import nltk
from collections import defaultdict


END_OF_SENTENCE_TAG = '<EOS>'
UNKNOWN_TAG = '<UNK>'
END_OF_PARAGRAPH_TAG = '<EOP>'
VOCAB_LOCATION = 'assets/vocab.json'
PADDING_TAG = 0

df = pd.read_csv('data/movies_genres.csv', sep='\t')


""" 
  dit is een zin. met nog, een zinnetje; plus dit. -> dit is een zin . met nog , een zinnetje ; plus dit .
"""

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
  vocab[END_OF_SENTENCE_TAG] = 1
  vocab[END_OF_PARAGRAPH_TAG] = 2

  for _, row in df.iterrows():
    tokenized = nltk.word_tokenize(row['plot'])
    for word, index in zip(tokenized, range(len(tokenized))):
      transformed_word = map_word(word, index, len(tokenized))
      if (transformed_word not in vocab):
        vocab[transformed_word] = len(vocab)

  return vocab


try:
  vocab = json.load(open(VOCAB_LOCATION))
except FileNotFoundError:
  vocab = create_vocab()
  with open(VOCAB_LOCATION, 'w') as f:
    json.dump(vocab, f)
  

def create_feature_vector(sample):
  tokenized = nltk.word_tokenize(sample)
  return torch.LongTensor([
    vocab[map_word(word, index, len(tokenized))]
    for word, index in zip(tokenized, range(len(tokenized)))
  ])


def load_batch(batch_size=5):
    sampled = df.sample(batch_size)

    features = [
        create_feature_vector(sample['plot'])
        for _, sample in sampled.iterrows()
    ]

    return torch.nn.utils.rnn.pad_sequence(features, batch_first=True)

VOCAB_SIZE = len(vocab)

if __name__ == "__main__":
  i = 0
  for batch in load_batch():
    i = i + 1
    print(batch)

    if i == 1:
      break
