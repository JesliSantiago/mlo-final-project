# poem classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import nltk
import string
import pickle

class poem_classifier():
    def __init__(self, model_path, tokenizer_path, maxlen):
        self.model = keras.models.load_model(model_path)
        self.maxlen = maxlen
        self._labels = ['Affection', 'Death', 'Environment', 'Music']
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def predict_(self, text):
        vectorized = self.tokenizer.texts_to_sequences(text)
        # print(vectorized)
        padded = tf.keras.utils.pad_sequences(vectorized, padding='pre', 
                                                 maxlen=self.maxlen)
        # print(padded)
        _preds = np.sum(self.model.predict(padded), axis=0)
        # print(np.argmax(_preds))
        preds = self._labels[np.argmax(_preds)]

        print(f"Predictions are {preds}")
        return preds