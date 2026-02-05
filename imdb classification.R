
# IMDB  (Binary Classification)
# Classifying movie reviews as positive or negetive (binary classification)

# Loading the keras library which is used for deep learning
library(keras3)  

# loading the Imdb dataset Load IMDB dataset with only the first 10,000 words
imdb <- dataset_imdb(num_words = 10000)

# splitting into train / test data and labels
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

# checking the structure of the trained data and labels
str(train_data)
str(train_labels)

# finding the maximum word index 
max(sapply(train_data, max))

# getting the word index mapping from word to integer 
word_index <- dataset_imdb_word_index()

# performing a reverse word index mapping from integer to word 
reverse_word_index <- names(word_index)
names(reverse_word_index) <- as.character(word_index)
