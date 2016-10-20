# -*- coding: utf-8 -*-
"""
An impementation of LIME (http://arxiv.org/abs/1602.04938), an algorithm to
explain predictions of black-box models.

Input:

1. a black-box classifier (or regressor - not implemented here).
2. an example to explain.

Main idea:

1. Generate a fake dataset from the example to explain: generated instances
   are changed versions of the example (e.g. for text it could be the same
   text, but with some words removed); generated labels are black-box
   classifier predictions for these generated examples.

2. Train a white-box classifer on these examples (e.g. a linear model).

   It helps if a white-box classifier takes in account how are examples
   changed in (1), e.g. uses a similar tokenization scheme.

   XXX: in the original lime code they use linear regression models trained on
   probability output; here we're using a classifier.

3. Explain the example through weights of this white-box classifier instead.

4. Prediction quality of a white-box classifer shows how well it approximates
   the black-box classifier. If the quality is low then probably explanation
   shouldn't be trusted (idea: choose number of examples generated at stage (1)
   automatically, based on a learning curve?).

   XXX: In the original lime code prediction quality is measured on the
   same data used for training; maybe it makes more sense to generate other
   examples for testing in order to prevent overfitting. Generated examples
   could be too close to each other though, so maybe it doesn't matter.

Tweaks:

1. Do feature selection for a white-box classifer in order to make it
   simpler and more explainable.
2. Weight generated examples by their distance to the original example:
   the further the generated example from the original example, the less
   it contributes to the loss function (sample_weights computed based
   on distance from the example to explain).

Even though the method is classifier-agnostic, it must make assumptions
about the kind of features the pipeline extracts from raw data.
The white-box classifer LIME uses is not required to use the same features
as black-box classifier, but a mistmatch between them limits explanation
quality.
"""
from __future__ import absolute_import
from typing import Any, Callable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
try:
    from sklearn.model_selection import train_test_split
except ImportError:  # sklearn < 0.18
    from sklearn.cross_validation import train_test_split

from eli5.lime import textutils
from eli5.lime.samplers import BaseSampler, MaskingTextSampler
from eli5.lime.utils import rbf, fit_proba


def _train_local_classifier(estimator,
                            samples,
                            similarity,
                            predict_proba,
                            expand_factor=10,
                            test_size=0.3,
                            ):
    # type: (Any, Any, np.ndarray, Callable[[Any], np.ndarray], int, float) -> float
    y_proba = predict_proba(samples)
    y_best = y_proba.argmax(axis=1)

    (X_train, X_test,
     y_proba_train, y_proba_test,
     y_best_train, y_best_test) = train_test_split(samples, y_proba, y_best,
                                                   test_size=test_size)

    # XXX: in the original lime code instead of a probabilitsic classifier
    # they build several regression models which try to output probabilities.
    #
    # XXX: Probability information is helpful because it could be hard
    # to get enough examples of all classes automatically, so we're fitting
    # classifier to produce the same probabilities, not only the same
    # best answer.

    # TODO: feature selection
    fit_proba(estimator, X_train, y_proba_train,
              expand_factor=expand_factor,
              sample_weight=similarity)

    # TODO/FIXME: score should take probabilities in account
    return estimator.score(X_test, y_best_test)


def get_local_pipeline_text(text, predict_proba, n_samples=1000,
                            expand_factor=10):
    """
    Train a classifier which approximates probabilistic text classifier locally.
    Return (clf, vec, score) tuple with "easy" classifier, "easy" vectorizer,
    and an estimated accuracy score of this pipeline, i.e.
    how well these "easy" vectorizer/classifier approximates text
    classifier in neighbourhood of ``text``.
    """
    vec = CountVectorizer(
        binary=True,
        token_pattern=textutils.DEFAULT_TOKEN_PATTERN,
    )
    clf = LogisticRegression(solver='lbfgs')  # supports sample_weight
    pipe = make_pipeline(vec, clf)

    sampler = MaskingTextSampler(bow=True)
    samples, similarity = sampler.sample_near(text, n_samples=n_samples)

    score = _train_local_classifier(
        estimator=pipe,
        samples=samples,
        similarity=similarity,
        predict_proba=predict_proba,
        expand_factor=expand_factor,
    )
    return clf, vec, score
