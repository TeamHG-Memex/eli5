#!/usr/bin/env python3

"""Utility functions for working with an IMDB sentiment classifier."""

import numpy as np
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence


vocablen = 10000
maxlen = 128


word_index = imdb.get_word_index()
word_index = {k:v+3 for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<OOV>'] = 2
word_index['<UNUSED>'] = 3 # need this to be consistent with the vectorized IMDB dataset
word_index = {k:v for k, v in word_index.items() if v < vocablen}


reversed_word_index = {v:k for k, v in word_index.items()}


def prepare_train_test_dataset():
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocablen,
                                                          seed=113,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=3, # <-- this does not seem to work?
                                                          )
    np.load = np_load_old

    x_train = pad_sequences(x_train,
                            maxlen=maxlen,
                            padding='post',
                            truncating='post',
                            value=word_index['<PAD>']
                            )

    x_test = pad_sequences(x_test,
                           maxlen=maxlen,
                           padding='post',
                           truncating='post',
                           value=word_index['<PAD>']
                           )
    return (x_train, y_train), (x_test, y_test)


def vectorized_to_tokens(X):
    if not isinstance(X[0], (list, np.ndarray)):
        # add batch dim
        X = [X]
    tokens = []
    for x in X:
        tokenized = [reversed_word_index.get(num, '<OOV>') for num in x]
        tokens.append(tokenized)
    return tokens


def string_to_vectorized(s, pad=False):
    # this model does not require fixed length input
    tokens = text_to_word_sequence(s)
    tokens.insert(0, '<START>')
    tokens = [tokens]
    if pad:
        # TODO: more control over padding
        tokens = pad_sequences(tokens,
                               maxlen=maxlen,
                               dtype=object,
                               padding='post',
                               truncating='post',
                               value='<PAD>',
                               )
    doc = [[word_index.get(token, word_index['<OOV>']) for token in sample] for sample in tokens]
    doc = np.array(doc)
    tokens = np.array(tokens)
    return doc, tokens


def tokens_to_vectorized(tokens):
    if not isinstance(tokens[0], (list, np.ndarray)):
        tokens = [tokens]
    doc = []
    for sample in tokens:
        x = [word_index.get(token, word_index['<OOV>']) for token in sample]
        doc.append(x)
    return np.array(doc)