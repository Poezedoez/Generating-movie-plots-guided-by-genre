import argparse
import torch
from multiprocessing import cpu_count
from torch.utils.data import DataLoader

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
    num_workers=cpu_count(),
    pin_memory=torch.cuda.is_available()
  )
  elbo_loss = 0
  for i, batch in enumerate(data_loader):
    if model.training:
      optimizer.zero_grad()
    elbo = model(batch['input'].to(device), batch['target'].to(device), batch['length'])
    if model.training:
      elbo.backward()
      optimizer.step()
      model.kl_anneal_step()
    elbo_loss += elbo.item()
    print(f"[Batch {i}/{len(data_loader)}, type: {data_type}, elbo: {elbo.item()}]")
  return elbo_loss / (i+1)


def run_epoch(model, datasets, device, optimizer):
  """
  First, runs one training epoch. Then, if a validation dataset
  is given, runs evaluation.
  """
  model.train()
  train_elbo = epoch_iter(model, datasets, device, optimizer, 'train')

  # Optionally, run a validation epoch
  val_elbo = None
  if datasets.get('val'):
    model.eval()
    with torch.no_grad():
      val_elbo = epoch_iter(model, datasets, device, optimizer, 'val')
  
  return train_elbo, val_elbo


def main(ARGS, device):
  """
  Prepares the datasets for training, and optional, validation and
  testing. Then, initializes the VAE model and runs the training (/validation)
  process for a given number of epochs.
  """
  data_splits = ['train', 'val']
  datasets = {
    split: IMDB(ARGS.data_dir, split, ARGS.max_sequence_length, ARGS.min_word_occ)
      for split in data_splits
  }
  model = VAE(
    datasets['train'].vocab_size, ARGS.batch_size, device,
    kl_anneal_type=ARGS.kl_anneal_type, kl_anneal_x0=ARGS.kl_anneal_x0,
    kl_anneal_k=ARGS.kl_anneal_k,
  )
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters())

  for epoch in range(ARGS.epochs):
    elbos = run_epoch(model, datasets, device, optimizer)
    train_elbo, val_elbo = elbos
    print(f"[Epoch {epoch} train elbo: {train_elbo}, val_elbo: {val_elbo}]")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', default=1, type=int,
                      help='max number of epochs')
  parser.add_argument('--batch_size', default=32, type=int,
                      help='batch size')
  parser.add_argument('--device', default='cpu', type=str,
                      help='device')
  parser.add_argument('--data_dir', default='data', type=str,
                      help='directory where the movies genre data is stored')
  parser.add_argument('--max_sequence_length', default='100', type=int,
                      help='max allowed length of the sequence')
  parser.add_argument('--min_word_occ', default='3', type=int,
                      help='only add word to vocabulary if occurence higher than this value')
  
  parser.add_argument('-kl_af', '--kl_anneal_type', type=str, default='logistic')
  parser.add_argument('-kl_k', '--kl_anneal_k', type=float, default=0.0025)
  parser.add_argument('-kl_x0', '--kl_anneal_x0', type=int, default=2500)

  ARGS = parser.parse_args()
  device = torch.device(ARGS.device)
  main(ARGS, device)
