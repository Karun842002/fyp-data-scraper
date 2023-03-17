#Maximum number of characters in a sentence = 44

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
import keras.preprocessing.sequence as sequence
from keras.layers import Input, Embedding, Dropout, Bidirectional, LSTM, Dense
from keras.models import Model
from keras.losses import BinaryCrossentropy

tk = Tokenizer(num_words=1000)
df = pd.read_csv('./datasets/cleaned2.csv')
x_train, x_val, y_train, y_val = train_test_split(df['simple_sentence'], df['truth_value'], test_size=0.33, random_state=42)

tk.fit_on_texts(x_train)
tokenized_train = tk.texts_to_sequences(x_train)
x_train = tf.keras.preprocessing.sequence.pad_sequences(tokenized_train, maxlen=60)
tokenized_test = tk.texts_to_sequences(x_val)
x_val = tf.keras.preprocessing.sequence.pad_sequences(tokenized_test, maxlen=60)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(df['truth_value'].tolist())
y_train = np.array(label_tokenizer.texts_to_sequences(y_train))
y_val = np.array(label_tokenizer.texts_to_sequences(y_val))

 
batch_size = 256
epochs = 25
embed_size = 60
input_layer = Input(shape=(60,))
embedding_layer = Embedding(input_dim=1000, output_dim=embed_size, input_length=2, trainable=False)(input_layer)
dropout_layer = Dropout(0.2)(embedding_layer)
bilstm_layer = Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.25, dropout=0.25))(dropout_layer)
bilstm_layer = Bidirectional(LSTM(units=64, return_sequences=True, recurrent_dropout=0.1, dropout=0.1))(bilstm_layer)
output_layer = Dense(units=1, activation='sigmoid')(bilstm_layer)
 
model = Model(inputs=[input_layer], outputs=[output_layer])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=BinaryCrossentropy(), metrics=['accuracy'])
 
model.summary()
history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, shuffle=True, verbose = 1)
