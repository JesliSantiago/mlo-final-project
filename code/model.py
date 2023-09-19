import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import nltk
import string
import pickle

class poem_classifier_model():
    def __init__(self):
        self.df_train = None
        self.df_test = None
        self.trained_model = None
        self.tokenizer = None
        self.max_len = None
        self.le = None
        self.otpt_act = tf.nn.softmax
        self.ls = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model = None
        self.good_model = None
        self.acc = tf.keras.metrics.Accuracy()
        self.prec = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        self.f1 = tf.keras.metrics.F1Score()

    def load_data(self, train=None, test=None):
        if train is None and test is None:
            self.df_train = pd.read_csv('https://raw.githubusercontent.com/brandynewanek/data/main/Poem_classification%20-%20train_data.csv')
            self.df_test = pd.read_csv('https://raw.githubusercontent.com/brandynewanek/data/main/Poem_classification%20-%20test_data.csv')
            # self.prec = tf.keras.metrics.Precision(num_classes=4)
            # self.recall = tf.keras.metrics.Recall(num_classes=4)
            # self.f1 = tf.keras.metrics.F1Score(num_classes=4)
        elif train is not None and test is not None:
            self.df_train = train
            self.df_test = test
        else:
            print("Incomplete input. Load failed.")

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

        counts = self.df_train['Genre'].value_counts()
        PCC = sum([(i/len(self.df_train))**2 for i in counts])
        self.thresh = 1.25*PCC

    def _tokenize_encode(self, inpt_tr, inpt_ts, otpt_tr, otpt_ts, num_words=500):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words,
                                                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~',
                                                          oov_token='<oov>')
        self.tokenizer.fit_on_texts(inpt_tr)
        inpt_tr = self.tokenizer.texts_to_sequences(inpt_tr)
        inpt_ts = self.tokenizer.texts_to_sequences(inpt_ts)
        self.max_len = max([len(s) for s in inpt_tr])
        inpt_tr = tf.keras.utils.pad_sequences(inpt_tr, padding='pre',
                                               maxlen=self.max_len)
        inpt_ts = tf.keras.utils.pad_sequences(inpt_ts, padding='pre',
                                               maxlen=self.max_len)
        self.le = LabelEncoder().fit(otpt_tr)
        otpt_tr = self.le.transform(otpt_tr)
        otpt_ts = self.le.transform(otpt_ts)
        return inpt_tr, inpt_ts, otpt_tr, otpt_ts
    
    def _is_good(self):
        if self.trained_model is not None:
            for val_acc in self.trained_model.history["val_acc"]:
                if val_acc > self.thresh:
                    self.good_model = True

    def _save(self, emb, lr, optim):
        self.model.save(f"../models/{emb}_{lr:.2f}_{optim}_acc_{max(self.trained_model.history['val_acc'])}.keras")
        with open(f"../tokenizers/{emb}_{lr:.2f}_{optim}_acc_{max(self.trained_model.history['val_acc'])}_maxlen_{self.max_len}.pickle", 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def train(self, num_words=500, embd_dim=512, lr=.001, epochs=16, optimizer='adam'):
        self.good_model = False
        inpt_tr, inpt_ts, otpt_tr, otpt_ts = train_test_split(self.df_train['Poem'], self.df_train['Genre'], 
                                                              test_size=.2)
        # print(inpt_ts)
        # print(otpt_ts)
        inpt_tr, inpt_ts, otpt_tr, otpt_ts = self._tokenize_encode(inpt_tr, inpt_ts, otpt_tr, otpt_ts)
        # print(otpt_ts)
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=lr)
        self.model = tf.keras.models.Sequential([tf.keras.layers.Embedding(num_words*3, embd_dim, 
                                                                      input_length=self.max_len),
                                            tf.keras.layers.SpatialDropout1D(.45),
                                            tf.keras.layers.GlobalAveragePooling1D(),
                                            tf.keras.layers.Flatten(),
                                            tf.keras.layers.Dense(4, activation=self.otpt_act)
                                            ])
        # accuracy for now
        # self.model.compile(loss=self.ls, optimizer=opt, metrics=[self.prec])
        self.model.compile(loss=self.ls, optimizer=opt, metrics=['acc'])
        # otpt_tr = tf.cast(otpt_tr, dtype=tf.float32)
        # print("train")
        # # print(inpt_tr, np.shape(inpt_tr))
        # print(np.shape(otpt_tr))
        # print("test")
        # # print(inpt_ts, np.shape(inpt_ts))
        # print(np.shape(otpt_ts))
        self.trained_model = self.model.fit(inpt_tr, otpt_tr, epochs=epochs, validation_data=(inpt_ts, otpt_ts))
        self._is_good()
        self._save(embd_dim, lr, optimizer)

    def test(self):
        if self.trained_model is not None:
            input = self.tokenizer.texts_to_sequences(self.df_test['Poem'])
            input = tf.keras.utils.pad_sequences(input, padding='pre', 
                                                 maxlen=self.max_len)
            output = self.le.transform(self.df_test['Genre'])
            results = self.model.evaluate(input, output)
            return results
        else:
            print("No trained model.")

    


