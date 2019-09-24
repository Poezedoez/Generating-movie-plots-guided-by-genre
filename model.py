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
    initial_cell = self.z_to_cell(z)
    decoded, _ = self.lstm(embedded, (initial_hidden, initial_cell))
    logits = self.lstm_to_vocab(decoded)
    probabilities = F.log_softmax(logits, dim=-1)

    return probabilities


class VAE(nn.Module):
  def __init__(self, vocab_size, batch_size, device, lstm_dim=100,
                z_dim=100, emb_dim=100):

    super(VAE, self).__init__()

    self.batch_size = batch_size
    self.device = device
    self.embedding = nn.Embedding(vocab_size, emb_dim).to(device)
    self.encoder = Encoder(vocab_size, lstm_dim, z_dim, emb_dim)
    self.decoder = Decoder(vocab_size, batch_size, lstm_dim, z_dim)
    # 0 = padding index
    self.NLL = nn.NLLLoss(size_average=False, ignore_index=0)

  def forward(self, x):
    max_seq_length = x.size()[1]
    embedded = self.embedding(x)
    mean, logvar = self.encoder(embedded)
    z = self.reparameterize(mean, logvar)
    logp = self.decoder(embedded, z)
    average_negative_elbo = self.elbo_loss_function(
      logp, x, max_seq_length, mean, logvar,
    )
    return average_negative_elbo

  def reparameterize(self, mean, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std, device=self.device)
    return mean + std*eps

  def elbo_loss_function(self, logp, target, max_seq_length, mean, logvar):
    target = target[:, :max_seq_length].contiguous().view(-1)
    logp = logp.view(-1, logp.size(2))

    # Negative log likelihood
    nll_loss = self.NLL(logp, target)

    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    # TODO: add KL annealing

    return (nll_loss + kl_loss) / self.batch_size
