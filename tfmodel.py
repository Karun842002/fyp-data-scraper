import tensorflow as tf
import tensorflow_addons as tfa

class BiLSTMCRF(tf.keras.Model):
    def __init__(self, vocab_size, num_tags, embedding_dim, lstm_units):
        super(BiLSTMCRF, self).__init__()
        
        # Initialize embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        # Initialize BiLSTM layer
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))
        
        # Initialize dense layer
        self.dense = tf.keras.layers.Dense(num_tags)
        
        # Initialize CRF layer
        self.crf = tfa.layers.CRF(num_tags)
        
    def call(self, inputs, training=None, mask=None):
        # Pass input sequence through embedding layer
        embeddings = self.embedding(inputs)
        
        # Pass embeddings through BiLSTM layer
        lstm_outputs = self.bilstm(embeddings)
        
        # Pass BiLSTM outputs through dense layer
        outputs = self.dense(lstm_outputs)
        
        # Pass dense layer outputs through CRF layer
        outputs = self.crf(outputs)
        
        return outputs


import pandas as pd
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

# Load the CSV file
df = pd.read_csv('text_dataset.csv')

# Convert text column to list
text_list = df['text_column'].tolist()

# Initialize tokenizer
tokenizer = Tokenizer()

# Fit tokenizer on the text data
tokenizer.fit_on_texts(text_list)

# Get the vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Get the maximum length of a sequence
max_length = max([len(s.split()) for s in text_list])

# Initialize the model
model = BiLSTMCRF(vocab_size, num_tags, embedding_dim, lstm_units)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(lr=0.001)

# Compile the model
model.compile(optimizer=optimizer, loss=model.crf.loss, metrics=[model.crf.accuracy])

# Train the model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=batch_size)
