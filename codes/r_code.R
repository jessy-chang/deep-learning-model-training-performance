library(keras)
library(data.table)
library(ggplot2)
library(caret)
library(reshape2)
# install_keras()

set.seed(3333)

# ------------------------------------------------------------------------
# User-defined functions
# ------------------------------------------------------------------------

# Show and store model results
model_results <- function(model, experiment, history, execution_time){
  results <- list()
  train_results <- model %>% evaluate(x_train, y_train, verbose = 0)
  test_results <- model %>% evaluate(x_test, y_test, verbose = 0)
  
  sink(paste0('Program Outputs/', experiment, '_performance.txt'))
  summary(model)
  
  cat(sprintf('\nProcessing time for training (seconds): %f',as.numeric(execution_time)*60))
  cat(sprintf('\nTraining set accuracy: %.4f', train_results$acc))
  cat(sprintf('\nTraining set loss: %.4f', train_results$loss))
  cat(sprintf('\nHold-out test set set accuracy: %.4f', test_results$acc))
  cat(sprintf('\nHold-out test set set loss: %.4f', test_results$loss))
  
  sink()
  
  results$history <- history
  results$process_time <- execution_time
  results$train_loss <- train_results$loss
  results$train_acc <- train_results$acc
  results$test_loss <- test_results$loss
  results$test_acc <- test_results$acc
  results$test_pred <- model %>% predict_classes(x_test)
  
  results
}   

# Plot confusion matrix
plot_confusion <- function(test_pred, experiment){
  cm <- confusionMatrix(factor(test_pred), factor(y_test),
                        dnn = c("Predict", "True"))
  melttb <- melt(cm$table)
  
  ggplot(data = melttb, aes(x=factor(Predict), y=factor(True))) + 
    geom_tile(aes(fill=value)) +
    scale_y_discrete(limits = rev(levels(factor(melttb$True)))) +
    geom_text(aes(label = sprintf("%1.0f", value)), vjust = 1) +
    scale_fill_gradient2() +
    xlab('Predicted Image') + 
    ylab('True Image') +
    theme(legend.position = "none")
  
  ggsave(paste0('Program Outputs/', experiment,'_fig_confusion_matrix.pdf'))
  
}


# ------------------------------------------------------------------------
# Model 1B
# ------------------------------------------------------------------------
cifar10 <- dataset_cifar10()
str(cifar10$train)

x_train <- cifar10$train$x/255
x_test <- cifar10$test$x/255
y_train <- cifar10$train$y
y_test <- cifar10$test$y


# Defining Model ----------------------------------------------------------

# Set model to run for epochs unless early stopping rule is met
max_epochs <- 100  
earlystop_callback <- callback_early_stopping(monitor = "val_acc", min_delta = 0.01,
                                             patience = 5, verbose = 0, mode = "auto",
                                             baseline = NULL, restore_best_weights = FALSE)

# Initialize sequential model
model1B <- keras_model_sequential()

model1B %>%
  
  # Start with hidden 2D convolutional layer being fed 32x32 pixel images
  layer_conv_2d(
    filter = 32, kernel_size = c(3,3), padding = "same", 
    input_shape = c(32, 32, 3)) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # 2 additional hidden 2D convolutional layers
  layer_conv_2d(filter = 64, kernel_size = c(3,3), padding = "same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter = 64, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  
  # Use max pooling once more
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto 10 unit output layer
  layer_dense(10) %>%
  layer_activation("softmax")

# Print model structure
summary(model1B)

# Compile model
model1B %>% compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics='accuracy')

# Training ----------------------------------------------------------------
start_time <- Sys.time()
history1B <- model1B %>% fit(x_train, y_train,
                             epochs = max_epochs,
                             validation_split = 0.2,
                             shuffle = FALSE,
                             verbose = 1,
                             callbacks = earlystop_callback)
end_time <- Sys.time()

execution_time1B <- end_time - start_time
plot(history1B)
ggsave(paste0('Program Outputs/', 'model1B','_fig_training_process.pdf'))

model1B_results <- model_results(model1B, 'model1B', history1B, execution_time1B)
plot_confusion(model1B_results$test_pred, 'model1B')



# ------------------------------------------------------------------------
# Model 2B
# ------------------------------------------------------------------------
# Embedding
max_features = 20000
maxlen = 250
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30

# Set model to run for epochs unless early stopping rule is met
max_epochs <- 100  
earlystop_callback <- callback_early_stopping(monitor = "val_acc", min_delta = 0.01,
                                              patience = 5, verbose = 0, mode = "auto",
                                              baseline = NULL, restore_best_weights = FALSE)




imdb <- dataset_imdb(num_words = max_features)
# Keras load all data into a list with the following structure:
str(imdb)

# Pad the sequences to the same length
# This will convert our dataset into a matrix: each line is a review
# and each column a word on the sequence
# We pad the sequences with 0s to the left
x_train <- imdb$train$x %>%
  pad_sequences(maxlen = maxlen)
x_test <- imdb$test$x %>%
  pad_sequences(maxlen = maxlen)
y_train <- imdb$train$y
y_test <- imdb$test$y

# Defining Model ------------------------------------------------------

model2B <- keras_model_sequential()

model2B %>%
  layer_embedding(max_features, embedding_size, input_length = maxlen) %>%
  layer_dropout(0.25) %>%
  layer_conv_1d(
    filters, 
    kernel_size, 
    padding = "valid",
    activation = "relu",
    strides = 1
  ) %>%
  layer_max_pooling_1d(pool_size) %>%
  layer_lstm(lstm_output_size) %>%
  layer_dense(1) %>%
  layer_activation("sigmoid")

model2B %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy"
)

# Training ----------------------------------------------------------------
start_time <- Sys.time()
history2B <- model2B %>% fit(x_train, y_train,
                             epochs = max_epochs,
                             validation_split = 0.2,
                             verbose = 1,
                             callbacks = earlystop_callback)
end_time <- Sys.time()

execution_time2B <- end_time - start_time
plot(history2B)
ggsave(paste0('Program Outputs/', 'model2B','_fig_training_process.pdf'))

model2B_results <- model_results(model2B, 'model2B', history2B, execution_time2B)
model2B_pred <- ifelse(model2B_results$test_pred>0.5,1,0)
plot_confusion(model2B_pred, 'model2B')
