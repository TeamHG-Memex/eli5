# -*- coding: utf-8 -*-
from __future__ import print_function

import pytest

keras = pytest.importorskip('keras')
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
import numpy as np

import eli5


MULTICONV_IMDB_SENTIMENT_CLASSIFIER = 'tests/estimators/keras_multiconv_imdb_sentiment_classifier.h5'


# TODO: test show_prediction with a model
# would need approximate truth techniques
# samples can be just individual words, but watch model size - should be specifically for testing


def decode(doc, reversed_word_index):
    # TODO: return numpy array
    return [reversed_word_index.get(x, '?')  
                for vec in doc 
                for x in vec]


def vectorize(tokens, word_index):
    num_words = len(word_index)
    vec = [word_index['<START>']]
    for token in tokens:
        x = word_index.get(token, word_index['<UNK>'])
        if num_words <= x:
            x = word_index['<UNK>']
        vec.append(x)
    return vec


# FIXME: this is very verbose


@pytest.fixture(scope='module')
def imdb_sentiment_clf():
    model = keras.models.load_model(MULTICONV_IMDB_SENTIMENT_CLASSIFIER)
    print('Summary of classifier:')
    model.summary()
    maxlen = 500
    return model, maxlen


@pytest.fixture(scope='module')
def imdb_index():
    num_words = 10000
    word_index = imdb.get_word_index()

    # add special tokens
    # FIXME: probably need to retrain using this set up
    word_index = {k:(v+2) for k, v in word_index.items()}
    word_index['<PAD>'] = 0
    word_index['<START>'] = 1
    word_index['<UNK>'] = 2

    # set maximum word number
    word_index = {k: v for k, v in word_index.items() if v < num_words}
    reversed_word_index = {v:k for k, v in word_index.items()}
    return word_index, reversed_word_index


@pytest.fixture(scope='module')
def imdb_sample_input(imdb_index, imdb_sentiment_clf):
    word_index, reversed_word_index = imdb_index
    maxlen = imdb_sentiment_clf[1]

    sample = 'good story but bad acting'
    tokens = text_to_word_sequence(sample) # to tokens
    doc = vectorize(tokens, word_index)
    doc = np.expand_dims(doc, axis=0) # add batch dimension
    doc = pad_sequences(doc, maxlen=maxlen, 
                        padding='post', truncating='post',
                        value=word_index['<PAD>']
    )
    # update tokens
    tokens = decode(doc, reversed_word_index)
    # tokens, = tokens # FIXME

    print('Input:')
    print(sample)
    print(tokens)
    print(doc)
    return doc, tokens, sample


# TODO: test no relu and counterfactual (sentiment analysis positive and negative words)
# TODO: test with and without padding
def test_sentiment_classification(imdb_sentiment_clf, imdb_index, imdb_sample_input):
    model = imdb_sentiment_clf[0]
    word_index = imdb_index[0]
    doc, tokens = imdb_sample_input[:2]
    expl = eli5.explain_prediction(model, doc, tokens=tokens, 
                                    pad_value=word_index['<PAD>'],
                                    padding='post',
    )
    # assert_spans_value_close_to(1)
    # assert_spans_value_close_to(-1)
    # assert_spans_value_close_to(0)


# FIXME: smaller model sizes. Combine RNN + conv. Combine tasks (multiclass cover sentiment?)
# 1: token level + LSTM masked + sentiment analysis + non-fixed len
# 2: char level + conv + multiclass + fixed len


# TODO: test with a multiclass model


# TODO: test with multilabel model


# TODO: test char level model