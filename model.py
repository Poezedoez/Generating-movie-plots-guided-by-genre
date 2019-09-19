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
        self.lstm_to_vocab = nn.Linear(lstm_dim, vocab_size)
        self.batch_size = batch_size
        self.lstm_dim = lstm_dim

    def forward(self, embedded, z):
        hidden = self.z_to_hidden(z)
        # reshaped = hidden.view(1,  self.lstm_dim, self.batch_size)
        print("hidden", hidden.size())
        random = torch.rand_like(hidden)  # dummy variable
        decoded, _ = self.lstm(embedded, (hidden, random))
        return decoded


class VAE(nn.Module):
    def __init__(self, vocab_size, batch_size, device, lstm_dim=100,
                 z_dim=100, emb_dim=100):

        super(VAE, self).__init__()

        self.batch_size = batch_size
        self.device = device
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = Encoder(vocab_size, lstm_dim, z_dim, emb_dim)
        self.decoder = Decoder(vocab_size, batch_size, lstm_dim, z_dim)
        # 0 = padding index
        self.NNL = nn.NNLLoss(size_average=False, ignore_index=0)

    def forward(self, x, y, max_seq_length):
        embedded = self.embedding(x)
        mean, logvar = self.encoder(embedded)
        z = self.reparameterize(mean, logvar)
        logp, reconstruction = self.decoder(embedded, z)
        average_negative_elbo = self.elbo_loss_function(
            logp, y, max_seq_length, mean, logvar,
        )
        return average_negative_elbo

    def reparameterize(self, mean, logvar)
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std, device=self.device)
    return mean + std*eps

    def elbo_loss_function(self, logp, target, max_seq_length, mean, logvar):
        target = target[:, :torch.max(
            max_seq_length).data[0]].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative log likelihood
        NLL_loss = self.NLL(logp, target)

        # KL Divergence
        KL_LOSS = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        # TODO: add KL annealing

        return (NLL_LOSS + KL_loss) / self.batch_size
