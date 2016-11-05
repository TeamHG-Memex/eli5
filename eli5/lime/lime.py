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
from typing import Any, Callable, Dict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from eli5.lime import textutils
from eli5.lime.samplers import MaskingTextSampler
from eli5.lime.utils import (
    fit_proba,
    score_with_sample_weight,
    mean_kl_divergence
)


def _train_local_classifier(estimator,
                            samples,
                            similarity,
                            predict_proba,
                            expand_factor=10,
                            test_size=0.3,
                            random_state=None,
                            ):
    # type: (Any, Any, np.ndarray, Callable[[Any], np.ndarray], int, float, Any) -> Dict[str, float]
    rng = check_random_state(random_state)
    y_proba = predict_proba(samples)

    (X_train, X_test,
     similarity_train, similarity_test,
     y_proba_train, y_proba_test) = train_test_split(samples,
                                                     similarity,
                                                     y_proba,
                                                     test_size=test_size,
                                                     random_state=rng)

    # XXX: in the original lime code instead of a probabilitsic classifier
    # they build several regression models which try to output probabilities.
    #
    # XXX: Probability information is helpful because it could be hard
    # to get enough examples of all classes automatically, so we're fitting
    # classifier to produce the same probabilities, not only the same
    # best answer.

    # TODO: feature selection
    # Ideally, it should be supported as a Pipeline (i.e. user should
    # be able to configure it).
    fit_proba(estimator, X_train, y_proba_train,
              expand_factor=expand_factor,
              sample_weight=similarity_train,
              random_state=rng)

    y_proba_test_pred = estimator.predict_proba(X_test)
    return {
        'mean_KL_divergence': mean_kl_divergence(
            y_proba_test_pred,
            y_proba_test,
            sample_weight=similarity_test
        ),
        'score': score_with_sample_weight(estimator,
                                          X_test,
                                          y_proba_test.argmax(axis=1),
                                          sample_weight=similarity_test)
    }


def get_local_pipeline_text(text, predict_proba, n_samples=1000,
                            expand_factor=10, random_state=None):
    """
    Train a classifier which approximates probabilistic text classifier locally.
    Return (clf, vec, metrics) tuple with "easy" classifier, "easy" vectorizer,
    and an estimated metrics of this pipeline, i.e.
    how well these "easy" vectorizer/classifier approximates text
    classifier in neighbourhood of ``text``.
    """
    rng = check_random_state(random_state)
    vec = CountVectorizer(
        binary=True,
        token_pattern=textutils.DEFAULT_TOKEN_PATTERN,
    )
    clf = LogisticRegression(C=100, random_state=rng)
    pipe = make_pipeline(vec, clf)

    sampler = MaskingTextSampler(bow=True, random_state=rng)
    samples, similarity = sampler.sample_near(text, n_samples=n_samples)

    metrics = _train_local_classifier(
        estimator=pipe,
        samples=samples,
        similarity=similarity,
        predict_proba=predict_proba,
        expand_factor=expand_factor,
        random_state=rng,
    )
    return clf, vec, metrics
