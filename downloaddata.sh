mkdir data

wget https://github.com/davidsbatista/text-classification/raw/master/movies_genres.csv.bz2

bunzip2 movies_genres.csv.bz2

mv movies_genres.csv data/
