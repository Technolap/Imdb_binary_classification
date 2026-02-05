
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

# converting the train and test lables to numeric
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

# Building the neural network model
model <- keras_model_sequential() %>%
  layer_dense(16, activation = "relu") %>%  # intermediate layer with 16 units
  layer_dense(16, activation = "relu") %>%  # 2nd intermediate layer with 16 units
  layer_dense(1, activation = "sigmoid")    # layer 3 (output layer which gives the scalar prediction regarding the sentiment of the current review)

# compiling the model with loss function, optimizer and the metrics 
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

# setting aside a validation set from the training data 
x_val <- x_train[seq(10000), ]
partial_x_train <- x_train[-seq(10000), ]
y_val <- y_train[seq(10000)]
partial_y_train <- y_train[-seq(10000)]

# training the model with the fit function
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

# plotting the training and validation loss and accuracy 
str(history$metrics)
plot(history)

# converting history to a data frame 
history_df <- as.data.frame(history)
str(history_df)

# retraining the model from scratch
model <- keras_model_sequential() %>%
  layer_dense(16, activation = "relu") %>%
  layer_dense(16, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")

# compiling the model 
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

# training the model for 4 epochs 
model %>% fit(x_train, y_train, epochs = 4, batch_size = 512)

# evaluating the model on the test data 
results <- model %>% evaluate(x_test, y_test)
results

# finally using the trained model to generate predictions on the new data 
model %>% predict(x_test)

