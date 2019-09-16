import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
  def __init__(self, vocab_size, lstm_dim=100, z_dim=100, emb_dim=100):
    super().__init__()
    self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=lstm_dim, num_layers=1)
    self.lstm_to_mu = nn.Linear(lstm_dim, z_dim)
    self.lstm_to_sigma = nn.Linear(lstm_dim, z_dim)

  def forward(self, embedded):
    "x should be long tensor with indices of the words"
    super().__init__()
    lstm_output = self.lstm(embedded)
    mu = self.lstm_to_mu(lstm_output)
    std = self.lstm_to_sigma(lstm_output)
    return mu, std

class Decoder(nn.Module):
  def __init__(self, vocab_size, lstm_dim=100, z_dim=100, emb_dim=100):
    super().__init__()
    self.lstm = nn.LSTM(input_size=emb_dim, lstm_dim=lstm_dim, num_layers=1)
    self.z_to_hidden = nn.Linear(z_dim, lstm_dim)
    self.lstm_to_vocab = nn.Linear(in_feature=lstm_dim, vocab_size)
    
  def forward(self, embedded, z):
    super().__init__()
    hidden = self.z_to_hidden(z)
    decoded = self.lstm(embedded, hidden)


class VAE(nn.Module):
  def __init__(self, vocab_size, lstm_dim=100, z_dim=100, emb_dim=100):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, emb_dim)
    self.encoder = Encoder(vocab_size, lstm_dim, z_dim, emb_dim)
    self.decoder = Encoder(vocab_size, lstm_dim, z_dim)

  def forward(self, x):
    embedded = self.embedding(x)
    mu, std = self.encoder(embedded)
    epsilon = torch.randn_like(std)
    z = mean + std * epsilon
    reconstruction = self.decoder(embedded, z)
