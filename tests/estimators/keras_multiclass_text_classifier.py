#!/usr/bin/env python3

import os.path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence


maxlen = 3193


# TODO: want to download these files, i.e. S3 URL?
base_path = 'tests/estimators'
WORD_INDEX = os.path.join(base_path, 'word_index.csv')
LABELS_INDEX = os.path.join(base_path, 'labels_index.csv')
# 1k samples to not crash anything
# alternatively pass option to pandas to only read first 1k
X_DATA = os.path.join(base_path, 'x_data_small.csv')
Y_DATA = os.path.join(base_path, 'y_data_small.csv')


df = pd.read_csv(WORD_INDEX, index_col=0)
word_index = df.to_dict("dict")['0']

df = pd.read_csv(LABELS_INDEX, index_col=0)
labels_index = df.to_dict("dict")['0']

reversed_word_index = {v: k for k, v in word_index.items()}
reversed_labels_index = {v: k for k, v in labels_index.items()}


def prepare_train_test_dataset(pad=True, split=0.5):
    y = np.loadtxt(Y_DATA, dtype="int", delimiter=',')

    df = pd.read_csv(X_DATA, usecols=range(1, 5153))
    x = list(df.values)
    x = [arr[~np.isnan(arr)].astype('int').tolist() for arr in x]

    if pad:
        x = pad_sequences(x,
                          padding='post',
                          truncating='post',
                          value=word_index['<PAD>'],
                          maxlen=maxlen,
                          )
    if split is not None:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=42)
        return (x_train, x_test), (y_train, y_test)
    else:
        return x, y


def decode(X):
    if not isinstance(X[0], (list, np.ndarray)):
        # add batch dim
        X = [X]
    strings = []
    for x in X:
        tokenized = [reversed_word_index.get(num, '<OOV>') for num in x]
        strings.append(''.join(tokenized))
    return strings


def decode_output(y):
    idx = np.argmax(y)
    return reversed_labels_index[idx]


def tokenize(s):
    if not isinstance(s, list):
        s = [s]
    all_tokens = []
    for sample in s:
        tokens = list(' '.join(text_to_word_sequence(sample, lower=False)))
        all_tokens.append(tokens)
    return all_tokens


def numericalize(tokens, pad=True):
    if not isinstance(tokens[0], list):
        tokens = [tokens]
    doc = []
    for sample in tokens:
        x = [word_index.get(token, word_index['<OOV>']) for token in sample]
        doc.append(x)
    if pad:
        doc = pad_sequences(doc,
                           padding='post',
                           truncating='post',
                           value=word_index['<PAD>'],
                           maxlen=maxlen,
                           )
    return doc


def vector_to_tokens(vec):
    if not isinstance(vec[0], (list, np.ndarray)):
        vec = [vec]
    all_tokens = []
    for x in vec:
        tokens = [reversed_word_index.get(num, '<OOV>') for num in x]
        all_tokens.append(tokens)
    return all_tokens


def vectorize(s):
    tokens = tokenize(s)
    doc = numericalize(tokens)
    tokens = vector_to_tokens(doc)
    return doc, tokens