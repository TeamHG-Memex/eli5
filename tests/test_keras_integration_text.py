# -*- coding: utf-8 -*-
"""Test integration of Keras Grad-CAM explanation for text."""
from __future__ import print_function

import pytest
from pytest import approx

keras = pytest.importorskip('keras')
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
import numpy as np

import eli5
from .estimators.keras_sentiment_classifier import keras_sentiment_classifier
from .estimators.keras_multiclass_text_classifier import keras_multiclass_text_classifier

# model 1: ~100,000 parameters
# For training details see
# tests/estimators/keras_sentiment_classifier/keras_sentiment_classifier.ipynb
# or https://www.kaggle.com/tobalt/keras-text-model-sentiment
# Features:
# embedding -> masking -> bidirectional LSTM -> dense
# token level + sentiment (binary) classification + non-fixed length input
# trained in the keras.datasets.imdb
KERAS_SENTIMENT_CLASSIFIER = 'tests/estimators/keras_sentiment_classifier/keras_sentiment_classifier.h5'


@pytest.fixture(scope='module')
def sentiment_clf():
    model = keras.models.load_model(KERAS_SENTIMENT_CLASSIFIER)
    print('Summary of classifier:')
    model.summary()
    return model


@pytest.fixture(scope='module')
def sentiment_input():
    sample = "good and bad"
    doc, tokens = keras_sentiment_classifier.string_to_vectorized(sample)
    print('Input:', sample, doc, tokens, sep='\n')
    return doc, tokens


@pytest.fixture(scope='module')
def sentiment_input_all_pad():
    sample = ""
    doc, tokens = keras_sentiment_classifier.string_to_vectorized(sample, pad=True)
    print('Input:', sample, doc, tokens, sep='\n')
    return doc, tokens


# model 2: ~200,000 parameters
# For training details see
# tests/estimators/keras_multiclass_text_classifier/keras_multiclass_text_classifier.ipynb
# or https://www.kaggle.com/tobalt/keras-text-model-multiclass
# Features:
# embedding -> conv + pooling -> global pooling -> dense
# character level + multiple classes + fixed length input (3193)
# trained on consumer financial complaints dataset
# https://www.kaggle.com/cfpb/us-consumer-finance-complaints
KERAS_MULTICLASS_CLASSIFIER = 'tests/estimators/keras_multiclass_text_classifier/keras_multiclass_text_classifier.h5'


complaints_mortgage_idx = 2
complaints_credit_card_idx = 3


@pytest.fixture(scope='module')
def multiclass_clf():
    model = keras.models.load_model(KERAS_MULTICLASS_CLASSIFIER)
    print('Summary of classifier:')
    model.summary()
    return model


@pytest.fixture(scope='module')
def multiclass_input():
    sample = "mortgage interest and credit card"
    doc, tokens = keras_multiclass_text_classifier.string_to_vectorized(sample)
    print('Input:', sample, doc, tokens, sep='\n')
    return doc, tokens


def assert_near_zero(val):
    """
    0.1 -> False. 
    0.09 -> True.
    """
    assert val == approx(0, abs=0.09)


def get_docs_weighted_spans(expl):
    """Get document and spans from explanation object."""
    # TODO: hard-coded for only 1 target
    ws = expl.targets[0].weighted_spans.docs_weighted_spans[0]
    spans = ws.spans  # -> list of ('token', [(start,end)...], weight) tuples
    document = ws.document
    print('WeightedSpans:', spans, sep='\n')
    print('Document:', document, sep='\n')
    print('Document indices:')
    for (i, ch) in enumerate(document):
        print(i, ch)
    return spans, document


def span_in(span, start, end):
    """Check that span's indices are between start and end inclusive."""
    # FIXME: we assume that a span only contains 1 tuple for its indices
    span_start, span_end = span[1][0]
    return start <= span_start and span_end <= end


def total_weight(spans):
    """Sum all weights of a list of spans."""
    return sum([span[2] for span in spans])


def sum_weights_over_ranges(spans, ranges):
    """Sum weights of spans whose indices are in the list of ranges."""
    total = 0
    for (start, end) in ranges:
        spans_in_range = list(filter(lambda span: span_in(span, start, end), spans))
        w = total_weight(spans_in_range)
        print('Spans in range:', spans_in_range, sep='\n')
        print('Weight for range:', w, sep='\n')
        total += w
    return total


def assert_weights_over_spans(spans, positive, negative, neutral):
    if positive:
        pos = sum_weights_over_ranges(spans, positive)
        assert pos > 0
    if negative:
        neg = sum_weights_over_ranges(spans, negative)
        assert neg < 0
    if neutral:
        neu = sum_weights_over_ranges(spans, neutral)
        assert_near_zero(neu)


# positive, negative, neutral are lists of (start, end) tuples (inclusive)
# (indices into the document str, representing a "range")
# that indicate what kind of values the weights should have for the range
@pytest.mark.parametrize('relu, counterfactual, '
                         'positive, negative, neutral', [
    (True, False, [(8, 12)], [], [(0, 7), (13, 20)]),  # positive class
    (True, True, [(17, 20)], [], [(0, 17)]),  # negative class
    (False, False, [(8, 12)], [(17, 20)], [(0, 7), (13, 16)]),  # both classes
])
def test_sentiment_classification(sentiment_clf, 
                                  sentiment_input,
                                  relu,
                                  counterfactual,
                                  positive,
                                  negative,
                                  neutral,
                                  ):
    model = sentiment_clf
    doc, tokens = sentiment_input
    print('Explaining with relu={} and counterfactual={}'.format(relu, counterfactual))
    res = eli5.explain_prediction(model, doc, tokens=tokens, relu=relu, counterfactual=counterfactual)
    print(res)
    spans, document = get_docs_weighted_spans(res)
    assert_weights_over_spans(spans, positive, negative, neutral)


# padding should have no effect on prediction (neutral)
def test_padding_no_effect(sentiment_clf, sentiment_input_all_pad):
    model = sentiment_clf
    doc, tokens = sentiment_input_all_pad
    res = eli5.explain_prediction(model, doc, tokens=tokens)
    spans, document = get_docs_weighted_spans(res)
    neutral = [(0, len(document))]
    weight = sum_weights_over_ranges(spans, neutral)
    assert_near_zero(weight)


# should be able to explain dense and final RNN layers
def test_explain_1d_layer_text(sentiment_clf, sentiment_input_all_pad):
    model = sentiment_clf
    doc, tokens = sentiment_input_all_pad
    eli5.explain_prediction(model, doc, tokens=tokens, layer=-1)


# should be able to take tokens without batch dimension
def test_tokens_not_batched(sentiment_clf, sentiment_input_all_pad):
    model = sentiment_clf
    doc, tokens = sentiment_input_all_pad
    tokens, = tokens
    eli5.explain_prediction(model, doc, tokens=tokens, layer=-1)


# check that explain+format == show
def test_show_explanation(sentiment_clf, sentiment_input):
    model = sentiment_clf
    doc, tokens = sentiment_input
    res = eli5.explain_prediction(model, doc, tokens=tokens)
    formatted = eli5.format_as_html(res,
                                    force_weights=False,
                                    show=eli5.formatters.fields.WEIGHTS
                                    )  # -> rendered template (str)
    ipython = eli5.show_prediction(model, doc, tokens=tokens)  # -> display object
    ipython_html = ipython.data  # -> str
    assert formatted == ipython_html


@pytest.mark.parametrize('targets, positive, negative, neutral', [
    ([complaints_credit_card_idx], [(22, 32)], [(0, 16)], []),
    ([complaints_mortgage_idx], [(0, 16)], [(22, 32)], []),
])
def test_multiclass_classification(multiclass_clf, 
                                   multiclass_input,
                                   targets,
                                   positive,
                                   negative,
                                   neutral,
                                   ):
    model = multiclass_clf
    doc, tokens = multiclass_input
    res = eli5.explain_prediction(model,
                                  doc,
                                  tokens=tokens,
                                  targets=targets,
                                  relu=False,
                                  pad_token='<PAD>',
                                  )
    print(res)
    spans, document = get_docs_weighted_spans(res)
    assert_weights_over_spans(spans, positive, negative, neutral)