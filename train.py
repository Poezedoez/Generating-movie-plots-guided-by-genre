from data_handler import load_batch, VOCAB_SIZE
from model import VAE


def epoch_iter(model, data):
    elbo_loss = 0
    data_length = len(data)
    for _, batch in enumerate(data):
        batch.to(device)
        if model.training:
            optimizer.zero_grad()
        elbo = model(batch)
        if model.training:
            elbo.backward()
            optimizer.step()
        elbo_loss += elbo.item()
    return elbo_loss / data_length


def run_epoch(model, data, optimizer):
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    with torch.no_grad():
        val_elbo = epoch_iter(model, valdata, optimizer)
    
    return train_elbo, val_elbo


def main():
    data = load_data(ARGS.batch_size)
    model = VAE(VOCAB_SIZE, ARGS.batch_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
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
    device = torch.device(ARGS.device)

    os.makedirs('./results', exists_ok=True)

    main()
