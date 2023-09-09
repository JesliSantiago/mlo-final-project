import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

import nltk
import string

class poem_classifier_model():
    def __init__(self):
        self.df_train = None
        self.df_test = None
        self.trained_model = None

    def load_data(self):
        self.df_train = pd.read_csv('https://raw.githubusercontent.com/brandynewanek/data/main/Poem_classification%20-%20train_data.csv')
        self.df_test = pd.read_csv('https://raw.githubusercontent.com/brandynewanek/data/main/Poem_classification%20-%20test_data.csv')

    def preprocess(self):
        self.df_train = self.df_train.dropna()
        self.df_train['Poem'] = self.df_train['Poem'].replace(r'\xa0', r' ')
        self.df_train['Poem'] = self.df_train['Poem'].str.split().str.join(' ')
        self.df_train['Poem'] = self.df_train['Poem'].str.replace('Poem"What', 'Poem "What')
        self.df_train['Poem'] = self.df_train['Poem'].str.replace('shade.When', 'shade. When')
        self.df_train['Poem'] = self.df_train['Poem'].str.replace('afraid.Now', 'afraid. Now')
        self.df_train['Poem'] = self.df_train['Poem'].str.replace('still!When', 'still! When')
        self.df_train['Poem'] = self.df_train['Poem'].str.replace('afraid.Now,', 'afraid. Now,')
        self.df_train['Poem'] = self.df_train['Poem'].str.replace('Big Game.Bigger', 'Big Game. Bigger')
        self.df_train['Poem'] = self.df_train['Poem'].str.replace('contained,a', 'contained, a')
        self.df_train['Poem'] = self.df_train['Poem'].str.lower()

        nltk.download('stopwords')
        stp_wrds = nltk.corpus.stopwords.words('english')
        
        self.df_train['Poem'] = self.df_train['Poem'].replace (stp_wrds, '')
        pntn = string.punctuation
        self.df_train['Poem'] = self.df_train['Poem'].replace (pntn, ' ')

    def _tokenize_encode(self, inpt_tr, inpt_ts, otpt_tr, otpt_ts, num_words=500):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words,
                                                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~',
                                                          oov_token='<oov>')
        tokenizer.fit_on_texts(inpt_tr)
        inpt_tr = tokenizer.texts_to_sequences(inpt_tr)
        inpt_ts = tokenizer.texts_to_sequences(inpt_ts)
        max_len = max([len(s) for s in inpt_tr])
        inpt_tr = tf.keras.utils.pad_sequences(inpt_tr, padding='pre',
                                               maxlen=max_len)
        inpt_ts = tf.keras.utils.pad_sequences(inpt_ts, padding='pre',
                                               maxlen=max_len)
        le = LabelEncoder().fit(otpt_tr)
        otpt_tr = le.transform(otpt_tr)
        otpt_ts = le.transform(otpt_ts)
        return inpt_tr, inpt_ts, otpt_tr, otpt_ts, max_len

    def train(self, num_words=500, embd_dim=512, lr=.001, epochs=16, optimizer='adam'):
        inpt_tr, inpt_ts, otpt_tr, otpt_ts = train_test_split(self.df_train['Poem'], self.df_train['Genre'], 
                                                              test_size=.2)
        inpt_tr, inpt_ts, otpt_tr, otpt_ts, max_len = self._tokenize_encode(inpt_tr, inpt_ts, otpt_tr, otpt_ts)
        otpt_act = tf.nn.softmax
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=lr)
        ls = tf.keras.losses.SparseCategoricalCrossentropy()
        model = tf.keras.models.Sequential([tf.keras.layers.Embedding(num_words*3, embd_dim,
                                                                    input_length=max_len),
                                            tf.keras.layers.SpatialDropout1D(.45),
                                            tf.keras.layers.GlobalAveragePooling1D(),
                                            tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(4, activation=otpt_act)
                                            ])
        model.compile(loss= ls, optimizer=opt, metrics=['acc'])
        self.trained_model = model.fit(inpt_tr, otpt_tr, epochs=epochs, validation_data=(inpt_ts, otpt_ts))

