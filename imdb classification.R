
# IMDB  (Binary Classification)
# Classifying movie reviews as positive or negetive (binary classification)

# Loading the keras library which is used for deep learning
library(keras3)  

# loading the Imdb dataset Load IMDB dataset with only the first 10,000 words
imdb <- dataset_imdb(num_words = 10000)


