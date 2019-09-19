import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
  def __init__(self, vocab_size, lstm_dim=100, z_dim=100, emb_dim=100):
    super(Encoder, self).__init__()

    self.lstm = nn.LSTM(emb_dim, lstm_dim, num_layers=1, batch_first=True)
    self.lstm_to_mu = nn.Linear(lstm_dim, z_dim)
    self.lstm_to_sigma = nn.Linear(lstm_dim, z_dim)

  def forward(self, embedded):
    "x should be long tensor with indices of the words"
    hidden, ht_ct = self.lstm(embedded)
    ht, ct = ht_ct
    mu = self.lstm_to_mu(ht)
    std = self.lstm_to_sigma(ht)
    return mu, std

class Decoder(nn.Module):
  def __init__(self,
    vocab_size,
    batch_size,
    lstm_dim=100,
    z_dim=100,
    emb_dim=100
  ):
    super(Decoder, self).__init__()

    self.lstm = nn.LSTM(emb_dim, lstm_dim, num_layers=1, batch_first=True)
    self.z_to_hidden = nn.Linear(z_dim, lstm_dim)
    self.z_to_cell = nn.Linear(z_dim, lstm_dim)
    self.lstm_to_vocab = nn.Linear(lstm_dim, vocab_size)
    self.batch_size = batch_size
    self.lstm_dim = lstm_dim
    
  def forward(self, embedded, z):
    initial_hidden = self.z_to_hidden(z)
    initial_cell = self.z_to_cell
    decoded, _ = self.lstm(embedded, (initial_hidden, initial_cell))
    logits = lstm_to_vocab(decoded)
    probabilities = F.log_softmax(logits, dim=-1)

    return decoded


class VAE(nn.Module):
  def __init__(self, vocab_size, batch_size, lstm_dim=100, z_dim=100, emb_dim=100):
    super(VAE, self).__init__()

    self.embedding = nn.Embedding(vocab_size, emb_dim)
    self.encoder = Encoder(vocab_size, lstm_dim, z_dim, emb_dim)
    self.decoder = Decoder(vocab_size, batch_size, lstm_dim, z_dim)

  def forward(self, x):
    embedded = self.embedding(x)
    mean, std = self.encoder(embedded)
    epsilon = torch.randn_like(std)
    z = mean + std * epsilon
    reconstruction = self.decoder(embedded, z)
    return reconstruction
