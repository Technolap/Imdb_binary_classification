
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


# decoding the first review back into words 
decoded_words <- train_data[[1]] %>%
  sapply(function(i) {
    if (i > 3) reverse_word_index[[as.character(i - 3)]]
    else "?"  
  })

# combining the words into a readable review 
decoded_review <- paste0(decoded_words, collapse = " ")
cat(decoded_review, "\n")

# function for vectorizing sequences into binary group of words
vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- array(0, dim = c(length(sequences), dimension))
  for (i in seq_along(sequences)) {
    sequence <- sequences[[i]]
    for (j in sequence)
      results[i, j] <- 1 
  }
  results
}

# vectorizing the train and the test data 
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

# displaying the structure of the vectoried trained data 
str(x_train)