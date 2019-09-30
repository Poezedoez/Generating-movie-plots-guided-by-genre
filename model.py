import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


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
    logvar = self.lstm_to_sigma(ht)
    return mu, logvar


class Decoder(nn.Module):
  def __init__(self,
    vocab_size,
    batch_size,
    lstm_dim=100,
    z_dim=100,
    emb_dim=100,
  ):
    super(Decoder, self).__init__()

    self.emb_dim = emb_dim
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

    decoded_padded = rnn_utils.pad_packed_sequence(decoded, batch_first=True)[0].contiguous()

    logits = self.lstm_to_vocab(decoded_padded)
    logp = F.log_softmax(logits, dim=-1)

    return logp


class VAE(nn.Module):
  def __init__(
      self, vocab_size, batch_size, device,
      trainset,
      lstm_dim=100, z_dim=100, emb_dim=100,
      kl_anneal_type=None, kl_anneal_x0=None, kl_anneal_k=None,
      kl_fbits_lambda=None,
      word_keep_rate=0.5,
  ):

    super(VAE, self).__init__()

    self.batch_size = batch_size
    self.device = device

    self.step = 0
    self.kl_anneal_type = kl_anneal_type
    self.kl_anneal_x0 = kl_anneal_x0
    self.kl_anneal_k = kl_anneal_k

    self.kl_fbits_lambda = kl_fbits_lambda

    self.word_keep_rate = word_keep_rate

    self.trainset = trainset

    self.embedding = nn.Embedding(vocab_size, emb_dim).to(device)
    self.encoder = Encoder(vocab_size, lstm_dim, z_dim, emb_dim)
    self.decoder = Decoder(vocab_size, batch_size, lstm_dim, z_dim, emb_dim)
    # Ignore the padding when calculating the differences
    self.NLL = nn.NLLLoss(size_average=False, ignore_index=self.trainset.pad_idx)
    self.latent_size = z_dim

  def forward(self, input_seq, target_seq, lengths):
    """"""
    # Sort the input sequences by their sequence length in descending order.
    # We do this because rnn_utils.pack_padded_sequence expects sequences
    # sorted by length in a decreasing order.
    sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
    input_seq_sorted = input_seq[sorted_indices]
    # Perform word embedding on input sequences
    input_seq_embedded = self.embedding(input_seq_sorted)
    # Pack the padded input sequences
    input_seq_packed = rnn_utils.pack_padded_sequence(
      input_seq_embedded, sorted_lengths.data.tolist(), batch_first=True)

    mean, logvar = self.encoder(input_seq_packed)
    z = self.reparameterize(mean, logvar)
    # NOTE: use input_seq_packed OR ((maybe add dropout and) + embedding dropout -> pack sequence)
    # Drop words from the sequence on which the encoder model is conditioned by
    # setting them to UNK tokens. Only drop a fraction of words given by
    # the drop keep rate.
    decoder_input_seq_packed = input_seq_packed
    if self.word_keep_rate > 0:
      prob = torch.rand(input_seq.size(), device=self.device)
      prob[(input_seq.data - self.trainset.sos_idx) * (input_seq.data - self.trainset.pad_idx) == 0] = 1
      decoder_input_seq = input_seq.clone()
      decoder_input_seq[prob > self.word_keep_rate] = self.trainset.unk_idx
      decoder_input_seq_embedded = self.embedding(decoder_input_seq)
      decoder_input_seq_packed = rnn_utils.pack_padded_sequence(
        decoder_input_seq_embedded, sorted_lengths.data.tolist(), batch_first=True)
    logp = self.decoder(decoder_input_seq_packed, z)
    # Revert back the order of the input since we have previously
    # sorted the input in descending order based on their sequence length.
    # We need to set it back to the original order for when calculating
    # the ELBO loss.
    _, reversed_indices = torch.sort(sorted_indices)
    logp = logp[reversed_indices]
    average_negative_elbo = self.elbo_loss_function(
      logp, target_seq, lengths, mean, logvar,
    )
    return average_negative_elbo

  def reparameterize(self, mean, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std, device=self.device)
    return mean + std*eps

  def elbo_loss_function(self, logp, target, length, mean, logvar):
    # cut-off unnecessary padding from target, and flatten
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp.view(-1, logp.size(2))

    # Negative log likelihood
    nll_loss = self.NLL(logp, target)

    # KL Divergence
    kl_div = torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
    # KL free bits
    if self.kl_fbits_lambda is not None:
      kl_div[kl_div < self.kl_fbits_lambda] = self.kl_fbits_lambda
    kl_loss = -0.5 * torch.sum(kl_div)
    # KL annealing
    kl_weight = self.kl_anneal_function()

    batch_size = target.size(0)
    return (nll_loss + kl_loss * kl_weight) / batch_size

  def sample(self, p):
    return torch.distributions.Categorical(p).sample()

  def sentence_mapping(self, imdb, sequence):
    return " ".join([imdb.i2w[str(idx)] 
      for idx in sequence 
      if idx != imdb.unk_idx and idx != imdb.sos_idx and idx != imdb.eos_idx]) + "."

  def inference(self, imdb, max_seq_len=100, z=None):
        
    if z is None:
      z = torch.randn((1, self.latent_size), device=self.device)
    
    hidden = self.decoder.z_to_hidden(z).unsqueeze(1)
    cell = self.decoder.z_to_cell(z).unsqueeze(1)

    # initialize sequence
    sequence = [imdb.sos_idx] + [imdb.unk_idx for _ in range(max_seq_len - 1)] 

    for i in range(1, max_seq_len):
      emb = self.embedding(torch.LongTensor(sequence).to(self.device)).unsqueeze(0)

      decoded, hidden_cell = self.decoder.lstm(emb, (hidden, cell))
      hidden, cell = hidden_cell
      logp = F.softmax(self.decoder.lstm_to_vocab(decoded), dim=-1)
      predicted = self.sample(logp[:, i]).item()
      sequence[i] = predicted
      if predicted == imdb.eos_idx:
        return self.sentence_mapping(imdb, sequence)

    return self.sentence_mapping(imdb, sequence)
  
  def kl_anneal_step(self):
    self.step += 1
  
  def kl_anneal_function(self):
    """"""
    if self.kl_anneal_type == 'logistic':
      return float(1/(1+np.exp(-self.kl_anneal_k*(self.step-self.kl_anneal_x0))))
    elif self.kl_anneal_type == 'linear':
      return min(1, self.step/self.kl_anneal_x0)
    return 1
