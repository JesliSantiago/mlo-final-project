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

class poem_classifier_app():
    def __init__(self, model):
        self.df_train = None
        self.df_train = None

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

        # counts = self.df_train['Genre'].value_counts()
        # PCC = sum([(i/len(self.df_train))**2 for i in counts])
        # self.thresh = 1.25*PCC