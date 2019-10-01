mkdir data

wget https://github.com/davidsbatista/text-classification/raw/master/movies_genres.csv.bz2
wget http://nlp.stanford.edu/data/glove.6B.zip

bunzip2 movies_genres.csv.bz2
unzip glove.6B.zip

mv movies_genres.csv data/
