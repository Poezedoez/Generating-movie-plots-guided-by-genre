# DL4NLP

#### Environment
In order to smooth version dependency hell, you can use this environment:
```conda create -f environment.yaml```

### Dataset
First, run `downloaddata.sh` to download the `movies_genres.csv` file into the `data` folder. The IMDB class inside `imdb.py` will split this data into separate train, validation, and test data files.

#### Training
Run train.py with desired arguments, example:
```python train.py --batch_size 32 --device cpu --max_batches_per_epoch 20```

#### To do
Condition movie plot generation on genre
