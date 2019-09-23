import argparse 

import torch
from data_handler import DataHandler
from model import VAE


def epoch_iter(model, data_handler, device, optimizer, data_type):
    elbo_loss = 0
    data_length = data_handler.get_data_length(data_type)
    for _ in range(data_handler.batches_in_epoch):
        batch = data_handler.load_batch(data_type)
        batch.to(device)
        if model.training:
            optimizer.zero_grad()
        elbo = model(batch)
        if model.training:
            elbo.backward()
            optimizer.step()
        elbo_loss += elbo.item()
    return elbo_loss / data_length

def run_epoch(model, data_handler, device, optimizer):
    model.train()
    train_elbo = epoch_iter(model, data_handler, device, optimizer, "train")

    model.eval()
    with torch.no_grad():
      val_elbo = epoch_iter(model, data_handler, device, optimizer, "val")
    
    return train_elbo, val_elbo

def main(ARGS, device):
  data_handler = DataHandler('data/movies_genres.csv', (0.8, 0.1, 0.1), 32)

  model = VAE(len(data_handler.vocab), ARGS.batch_size, device)
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters())

  for epoch in range(ARGS.epochs):
    elbos = run_epoch(model, data_handler, device, optimizer)
    train_elbo, val_elbo = elbos
    print(f"[Epoch {epoch} train elbo: {train_elbo}, val_elbo: {val_elbo}]")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', default=1, type=int,
                      help='max number of epochs')
  parser.add_argument('--batch_size', default=5, type=int,
                      help='batch size')
  parser.add_argument('--device', default='cpu', type=str,
                      help='device')

  ARGS = parser.parse_args()
  print(ARGS)
  device = torch.device(ARGS.device)
  main(ARGS, device)
