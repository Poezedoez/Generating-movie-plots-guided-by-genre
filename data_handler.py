import pandas as pd
import torch
import json
from collections import defaultdict


END_OF_SENTENCE_TAG = '<EOS>'
UNKNOWN_TAG = '<UNK>'
END_OF_PARAGRAPH_TAG = '<EOP>'
VOCAB_LOCATION = 'assets/vocab.json'

df = pd.read_csv('data/movies_genres.csv', sep='\t')

def transform_word(word):
  return word.lower()


def custom_split(seperators, string):
  result = []
  partial = ""
  for character in string:
    if character
    if character in seperators:
      result.append(partial)
      partial = ""
    else:
      partial += character

  return result


def create_vocab():
  # if word is unknown, return corresponding index of <UNK>
  vocab = defaultdict(lambda: 0)

  vocab[UNKNOWN_TAG] = 0
  vocab[END_OF_SENTENCE_TAG] = 1
  vocab[END_OF_PARAGRAPH_TAG] = 2

  for _, row in df.iterrows():
    for word in custom_split([' ', '.'], row['plot']):
      transformed_word = transform_word(word)
      if (transformed_word not in vocab):
        vocab[transformed_word] = len(vocab)

  return vocab


try:
  vocab = json.load(open(VOCAB_LOCATION))
except FileNotFoundError:
  vocab = create_vocab()
  with open(VOCAB_LOCATION, 'w') as f:
    json.dump(vocab, f)


def map_word(word, index, sample_length):
  if word == ".":
    return END_OF_SENTENCE_TAG

  if (index - 1) == sample_length:
    return END_OF_PARAGRAPH_TAG

  return word
  

def create_feature_vector(sample):
  return [
    vocab[map_word(word, index, len(sample))]
    for word, index in zip(sample, range(len(sample)))
  ]


def load_batch(batch_size=256):
    sampled = df.sample(batch_size)

    features = [
        create_feature_vector(sample)
        for sample in sampled
    ]

    print(features)

    yield torch.LongTensor(features)

if __name__ == "__main__":
  i = 0
  for batch in load_batch():
    i = i + 1
    print(batch)

    if i == 5:
      break
