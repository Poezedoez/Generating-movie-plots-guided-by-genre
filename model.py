import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import os
import pickle

class Encoder(nn.Module):
  def __init__(self, vocab_size, lstm_dim, z_dim, embed_dim):
    super(Encoder, self).__init__()

    self.lstm = nn.LSTM(embed_dim, lstm_dim, num_layers=1, batch_first=True)
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
    lstm_dim,
    z_dim,
    embed_dim,
  ):
    super(Decoder, self).__init__()

    self.embed_dim = embed_dim
    self.lstm = nn.LSTM(embed_dim, lstm_dim, num_layers=1, batch_first=True)
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
      pretrained_embeddings,
      trainset, max_sequence_length,
      lstm_dim, z_dim, embed_dim,
      kl_anneal_type=None, kl_anneal_x0=None, kl_anneal_k=None,
      kl_fbits_lambda=None,
      word_keep_rate=0.5,
  ):

    super(VAE, self).__init__()

    self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    self.batch_size = batch_size
    self.device = device

    self.step = 0
    self.kl_anneal_type = kl_anneal_type
    self.kl_anneal_x0 = kl_anneal_x0
    self.kl_anneal_k = kl_anneal_k

    self.kl_fbits_lambda = kl_fbits_lambda

    self.word_keep_rate = word_keep_rate

    self.trainset = trainset
    self.max_sequence_length = max_sequence_length

    # self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)
    self.embedding = nn.Embedding.from_pretrained(
      pretrained_embeddings, freeze=False).to(device)
    self.encoder = Encoder(vocab_size, lstm_dim, z_dim, embed_dim)
    self.decoder = Decoder(vocab_size, batch_size, lstm_dim, z_dim, embed_dim)
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

  @staticmethod
  def load(path):
    with open(path, 'rb') as f:
      return pickle.load(f)

  def save(self, path):
    with open(path, 'wb') as f:
      pickle.dump(self, f)

  def sample(self, p):
    return torch.distributions.Categorical(p).sample()

  def sentence_mapping(self, imdb, sequence):
    return " ".join([imdb.i2w[str(idx)] 
      for idx in sequence 
      if idx != imdb.unk_idx and idx != imdb.sos_idx and idx != imdb.eos_idx]) + "."

  # def inference(self, imdb, max_seq_len=100, z=None):
        
  #   if z is None:
  #     z = torch.randn((1, self.latent_size), device=self.device)
    
  #   print('inference', self.decoder.z_to_hidden(z).size())
  #   hidden = self.decoder.z_to_hidden(z).unsqueeze(1)
  #   print('hidden unsqueezed', hidden.size())
  #   cell = self.decoder.z_to_cell(z).unsqueeze(1)

  #   # initialize sequence
  #   sequence = [imdb.sos_idx] + [imdb.unk_idx for _ in range(max_seq_len - 1)] 

  #   for i in range(1, max_seq_len):
  #     emb = self.embedding(torch.LongTensor(sequence).to(self.device)).unsqueeze(0)

  #     decoded, hidden_cell = self.decoder.lstm(emb, (hidden, cell))
  #     hidden, cell = hidden_cell
  #     logp = F.softmax(self.decoder.lstm_to_vocab(decoded), dim=-1)
  #     predicted = self.sample(logp[:, i]).item()
  #     sequence[i] = predicted
  #     if predicted == imdb.eos_idx:
  #       return self.sentence_mapping(imdb, sequence)

  #   return self.sentence_mapping(imdb, sequence)
  
  def inference(self, n=4, z=None):
    if z is None:
      batch_size = n
      z = torch.randn((batch_size, self.latent_size), device=self.device)
    else:
      batch_size = z.size(0)
    
    hidden = self.decoder.z_to_hidden(z).unsqueeze(0)
    cell = self.decoder.z_to_cell(z).unsqueeze(0)

    # required for dynamic stopping of sentence generation
    sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
    sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
    sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

    running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

    generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.trainset.pad_idx).long()

    t = 0
    while (t<self.max_sequence_length and len(running_seqs)>0):
      if t == 0:
          input_sequence = torch.Tensor(batch_size).fill_(self.trainset.sos_idx).long()

      input_sequence = input_sequence.unsqueeze(1)

      input_embedded = self.embedding(input_sequence)

      decoded, (hidden, cell) = self.decoder.lstm(input_embedded, (hidden, cell))

      logits = self.decoder.lstm_to_vocab(decoded)

      input_sequence = self._sample(logits)

      # save next input
      generations = self._save_sample(generations, input_sequence, sequence_running, t)

      # update gloabl running sequence
      sequence_mask[sequence_running] = (input_sequence != self.trainset.eos_idx).data
      sequence_running = sequence_idx.masked_select(sequence_mask)

      # update local running sequences
      running_mask = (input_sequence != self.trainset.eos_idx).data
      running_seqs = running_seqs.masked_select(running_mask)

      # prune input and hidden state according to local update
      if len(running_seqs) > 0:
        input_sequence = input_sequence[running_seqs]
        hidden = hidden[:, running_seqs]
        cell = cell[:, running_seqs]

        running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

      t += 1
    
    return generations

  def _sample(self, dist, greedy_mode=False):
    if greedy_mode:
      _, sample = torch.topk(dist, 1, dim=-1)
    else:
      probs = F.softmax(dist, dim=-1)
      sample = torch.distributions.Categorical(probs).sample()
    sample = sample.flatten()
    return sample

  def _save_sample(self, save_to, sample, running_seqs, t):
    # select only still running
    running_latest = save_to[running_seqs]
    # update token at position t
    running_latest[:,t] = sample.data
    # save back
    save_to[running_seqs] = running_latest
    return save_to

  def kl_anneal_step(self):
    self.step += 1
  
  def kl_anneal_function(self):
    """"""
    if self.kl_anneal_type == 'logistic':
      return float(1/(1+np.exp(-self.kl_anneal_k*(self.step-self.kl_anneal_x0))))
    elif self.kl_anneal_type == 'linear':
      return min(1, self.step/self.kl_anneal_x0)
    return 1
