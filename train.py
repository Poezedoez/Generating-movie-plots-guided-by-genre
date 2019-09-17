from data_handler import load_batch, VOCAB_SIZE
from model import VAE

BATCH_SIZE = 5
EPOCH_AMOUNT = 1

model = VAE(VOCAB_SIZE, BATCH_SIZE)

for i in range(EPOCH_AMOUNT):
    batch = load_batch(batch_size=BATCH_SIZE)
    print(model(batch))
