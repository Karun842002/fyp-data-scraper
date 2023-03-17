import datetime
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

class BiLSTMCRF(tf.keras.Model):
    def __init__(self, vocab_size, num_tags, embedding_dim, lstm_units):
        super(BiLSTMCRF, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))
        self.dense = tf.keras.layers.Dense(num_tags)
        # self.crf = tfa.layers.CRF(num_tags)
        
    def call(self, inputs, training=None, mask=None):
        embeddings = self.embedding(inputs)
        lstm_outputs = self.bilstm(embeddings)
        outputs = self.dense(lstm_outputs)
        # outputs = self.crf(outputs)
        return outputs

df = pd.read_csv('./datasets/cleaned2.csv')
text_list = df['simple_sentence'].tolist()
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(text_list)
vocab_size = len(tokenizer.word_index) + 1
max_length = max([len(s.split()) for s in text_list])
word_index = tokenizer.word_index

x_train, x_val, y_train, y_val = train_test_split(df['simple_sentence'], df['truth_value'], test_size=0.33, random_state=42)


train_sequences = tokenizer.texts_to_sequences(x_train)
val_sequences = tokenizer.texts_to_sequences(x_val)
train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
validation_padded = tf.keras.preprocessing.sequence.pad_sequences(val_sequences, maxlen=max_length, padding='post', truncating='post')
label_tokenizer = tf.keras.preprocessing.text.Tokenizer()
label_tokenizer.fit_on_texts(df['truth_value'].tolist())

training_label_seq = np.array(label_tokenizer.texts_to_sequences(y_train))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(y_val))
embedding_dim = 100
num_tags = 2 + 1
lstm_units = 4
num_epochs = 1

dataset = tf.data.experimental.make_csv_dataset('./datasets/cleaned2.csv', batch_size=32, num_epochs=1, label_name='truth_value', ignore_errors=True)


# log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
with tf.device("/cpu:0"):
    model = BiLSTMCRF(vocab_size, num_tags, embedding_dim, lstm_units)
    model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics= ['accuracy', 'loss'])
    history = model.fit(dataset, epochs=num_epochs, verbose=2)

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")