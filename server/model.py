import os
import torch
import torch.nn as nn
from keras.preprocessing import text
import tensorflow as tf
import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from simplifier_new import serverParser
import numpy as np
import pickle

class BiLSTM2(nn.Module):
    def __init__(self, num_words, embed_size, hidden_size, fc_out_size, output_size, dropout_rate):
        super(BiLSTM2, self).__init__()
        self.embedding = nn.Embedding(num_words, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.bilstm1 = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.bilstm1(x)
        x = self.dropout(x)
        x, _ = self.bilstm2(x)
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])
        return self.sigmoid(x)

class Model:
    def __init__(self, static_folder):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        self.tk1 = text.Tokenizer(num_words=2000)
        self.tk2 = text.Tokenizer(num_words=2000)
        self.X, self.y = self.preprocess(os.path.join(static_folder, 'dataset.csv')) 
        self.tk1.fit_on_texts(self.X['claim'])
        self.tk2.fit_on_texts(self.X['simple_sentence'])
        self.m1 = BiLSTM2(num_words=2000, embed_size=60, hidden_size=64, fc_out_size=5000, output_size=1, dropout_rate=0.2)
        self.m2 = BiLSTM2(num_words=2000, embed_size=60, hidden_size=64, fc_out_size=5000, output_size=1, dropout_rate=0.2)
        s1 = torch.load(os.path.join(static_folder,'raw.pt'), map_location=torch.device('cpu'))
        s2 = torch.load(os.path.join(static_folder,'ss.pt'), map_location=torch.device('cpu'))
        self.m1.load_state_dict(s1)
        self.m2.load_state_dict(s2)
        self.clf = pickle.load(open(os.path.join(static_folder, 'svm.pkl'), 'rb'))

    def tokenizeSentence(self, sentence, simple_sentence):
        tokenized_claim = self.tk1.texts_to_sequences([sentence, ])
        tokenized_ss = self.tk2.texts_to_sequences([simple_sentence, ])
        X_claim = torch.tensor(tf.keras.preprocessing.sequence.pad_sequences(tokenized_claim, maxlen=60))
        X_ss = torch.tensor(tf.keras.preprocessing.sequence.pad_sequences(tokenized_ss, maxlen=60))
        return X_claim, X_ss

    def preprocess(self, dataset_path):
        le = LabelEncoder()
        lemmatizer = nltk.stem.WordNetLemmatizer()

        def lemmatize_text(text):
            return ' '.join([lemmatizer.lemmatize(w) for w in text.split(' ')])

        df = pd.read_csv(dataset_path, header=0)
        df['truth_value'] = le.fit_transform(df['truth_value'])
        
        def isfloat(num):
            try:
                float(num)
                return True
            except ValueError:
                return False
        
        T = df['claim'].str.split(' \n\n---\n\n').str[0]
        T = T.str.replace('-',' ').str.replace('[^\w\s]','').str.replace('\n',' ').str.lower()
        stop = stopwords.words('english')
        T = T.apply(lambda x: ' '.join([y for y in str(x).split() if not (y.isdigit() or isfloat(y))]))
        T = T.apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))
        df['claim'] = T

        T = df['simple_sentence'].str.split(' \n\n---\n\n').str[0]
        T = T.str.replace('-',' ').str.replace('[^\w\s]','').str.replace('\n',' ').str.lower()
        stop = stopwords.words('english')
        T = T.apply(lambda x: ' '.join([y for y in str(x).split() if not (y.isdigit() or isfloat(y))]))
        T = T.apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))
        df['simple_sentence'] = T
        X = df[['claim', 'simple_sentence']]
        y = df['truth_value']

        X['claim'] = X['claim'].apply(lambda w: lemmatize_text(w))
        X['simple_sentence'] = X['simple_sentence'].apply(lambda w: lemmatize_text(w))

        return X, y
    
    def predict(self, sentence):
        simple_sentence = serverParser(sentence)
        X_claim, X_ss = self.tokenizeSentence(sentence, simple_sentence)
        p1 = self.m1(torch.tensor(np.array(X_claim)))
        p1 = (p1 > 0.5).to('cpu').int().squeeze().numpy().tolist()
        p2 = self.m2(torch.tensor(np.array(X_ss)))
        p2 = (p2 > 0.5).to('cpu').int().squeeze().numpy().tolist()
        svmx = [X_claim[0].detach().cpu().numpy().tolist() + [p1, p2], ]
        p = np.array(self.clf.predict_proba(svmx)) 
        prob = p[0][1] * 100
        return prob