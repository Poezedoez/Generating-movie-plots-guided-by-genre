import argparse
import torch
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
import os
import numpy as np

from imdb import IMDB
from model import VAE


def epoch_iter(model, datasets, device, optimizer, data_type):
  """
  Perform a single epoch for either the training or validation.
  use model.training to determine if in 'training mode' or not.
  
  Returns the average elbo for the complete epoch.
  """
  data_loader = DataLoader(
    dataset=datasets[data_type],
    batch_size=ARGS.batch_size,
    shuffle=data_type=='train',
    # num_workers=cpu_count(),
    num_workers=1,
    # pin_memory=torch.cuda.is_available()
  )
  elbo_loss = 0

  for i, batch in enumerate(data_loader):
    if i > ARGS.max_batches_per_epoch:
      break

    if model.training:
      optimizer.zero_grad()
    elbo = model(batch['input'], batch['target'], batch['length'])
    if model.training:
      elbo.backward()
      optimizer.step()
      model.kl_anneal_step()
    elbo_loss += elbo.item()
  return elbo_loss / (i+1)


def run_epoch(model, datasets, device, optimizer):
  """
  First, runs one training epoch. Then, if a validation dataset
  is given, runs evaluation.
  """
  model.train()
  train_elbo = epoch_iter(model, datasets, device, optimizer, 'train')

  if torch.cuda.is_available():
    torch.cuda.empty_cache()

  # Optionally, run a validation epoch
  val_elbo = None
  if datasets.get('val'):
    model.eval()
    with torch.no_grad():
      val_elbo = epoch_iter(model, datasets, device, optimizer, 'val')
  
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  
  return train_elbo, val_elbo


def main(ARGS, device):
  """
  Prepares the datasets for training, and optional, validation and
  testing. Then, initializes the VAE model and runs the training (/validation)
  process for a given number of epochs.
  """
  data_splits = ['train', 'val']
  datasets = {
    split: IMDB(ARGS.data_dir, split, ARGS.max_sequence_length, ARGS.min_word_occ,
        ARGS.create_data)
      for split in data_splits
  }
  pretrained_embeddings = datasets['train'].get_pretrained_embeddings(
    ARGS.embed_dim).to(device)
  model = VAE(
    datasets['train'].vocab_size, ARGS.batch_size, device,
    pretrained_embeddings=pretrained_embeddings,
    trainset=datasets['train'],
    max_sequence_length=ARGS.max_sequence_length,
    lstm_dim=ARGS.lstm_dim, z_dim=ARGS.z_dim, embed_dim=ARGS.embed_dim,
    kl_anneal_type=ARGS.kl_anneal_type, kl_anneal_x0=ARGS.kl_anneal_x0,
    kl_anneal_k=ARGS.kl_anneal_k,
    kl_fbits_lambda=ARGS.kl_fbits_lambda,
    word_keep_rate=ARGS.word_keep_rate,
  )
  model.to(device)

  optimizer = torch.optim.Adam(model.parameters())

  print('Starting training process...')

  amount_of_files = len(os.listdir("trained_models"))
  for epoch in range(ARGS.epochs):
    elbos = run_epoch(model, datasets, device, optimizer)
    train_elbo, val_elbo = elbos
    print(f"[Epoch {epoch} train elbo: {train_elbo}, val_elbo: {val_elbo}]")

    # Perform inference on the trained model
    with torch.no_grad():
      model.eval()
      samples = model.inference()
      print(*idx2word(samples, i2w=datasets['train'].i2w, pad_idx=datasets['train'].pad_idx), sep='\n')

    model.save(f"trained_models/{amount_of_files + 1}.model")


def idx2word(idx, i2w, pad_idx):
  sent_str = [str()]*len(idx)
  for i, sent in enumerate(idx):
    for word_id in sent:
      if word_id == pad_idx:
        break
      sent_str[i] += i2w[str(word_id.item())] + " "
    sent_str[i] = sent_str[i].strip()
  return sent_str


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--create_data', default=False, type=bool,
                      help='whether to create new IMDB data files')

  parser.add_argument('--epochs', default=30, type=int,
                      help='max number of epochs')
  parser.add_argument('--batch_size', default=4, type=int,
                      help='batch size')
  parser.add_argument('--device', default='cpu', type=str,
                      help='device')
  parser.add_argument('--data_dir', default='data', type=str,
                      help='directory where the movies genre data is stored')
  parser.add_argument('--max_sequence_length', default=60, type=int,
                      help='max allowed length of the sequence')
  parser.add_argument('--min_word_occ', default='3', type=int,
                      help='only add word to vocabulary if occurence higher than this value')
  parser.add_argument('--max_batches_per_epoch', default=100, type=int,
                      help='only run one epoch for this number of batches')

  parser.add_argument('--embed_dim', type=int, default=300)
  parser.add_argument('--z_dim', type=int, default=16)
  parser.add_argument('--lstm_dim', type=int, default=256)
  
  parser.add_argument('-kl_at', '--kl_anneal_type', type=str, default='logistic')
  parser.add_argument('-kl_k', '--kl_anneal_k', type=float, default=0.0025)
  parser.add_argument('-kl_x0', '--kl_anneal_x0', type=int, default=4500)

  parser.add_argument('-kl_l', '--kl_fbits_lambda', type=float, default=5.0)

  parser.add_argument('-wkr', '--word_keep_rate', type=float, default=0.75)

  ARGS = parser.parse_args()
  device = torch.device(ARGS.device)
  main(ARGS, device)
