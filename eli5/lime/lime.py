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
from typing import Any, Callable, Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.base import clone, BaseEstimator

import eli5
from eli5.lime.samplers import BaseSampler
from eli5.lime.textutils import DEFAULT_TOKEN_PATTERN, CHAR_TOKEN_PATTERN
from eli5.lime.samplers import MaskingTextSamplers
from eli5.lime.utils import (
    fit_proba,
    score_with_sample_weight,
    mean_kl_divergence,
    fix_multiclass_predict_proba,
    rbf
)
from eli5.lime._vectorizer import SingleDocumentVectorizer


def _train_local_classifier(estimator,
                            samples,
                            similarity,        # type: np.ndarray
                            y_proba,           # type: np.ndarray
                            expand_factor=10,  # type: int
                            test_size=0.3,     # type: float
                            random_state=None,
                            ):
    # type: (...) -> Dict[str, float]
    rng = check_random_state(random_state)

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
    if y_proba_test_pred.shape != y_proba_test.shape:
        # Sometimes generated training labels may contain only a subset of
        # target classes; it means it could happen that dimensions
        # of predicted probability matrices don't match.
        #
        # FIXME: the fix is not complete; to explain predictions
        # of the fitted estimator one must take care of target_names.
        if not hasattr(estimator, 'classes_'):
            raise ValueError("Result dimensions don't match and estimator"
                             "doesn't provide 'classes_' attribute; can't"
                             "figure out how are columns related.")
        seen_classes = estimator.classes_
        complete_classes = list(range(y_proba.shape[1]))
        y_proba_test_pred = fix_multiclass_predict_proba(
            y_proba=y_proba_test_pred,
            seen_classes=seen_classes,
            complete_classes=complete_classes
        )

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


class TextExplainer(BaseEstimator):
    def __init__(self,
                 n_samples=5000,  # type: int
                 char_based=False,  # type: bool
                 clf=None,
                 vec=None,
                 sampler=None,  # type: BaseSampler
                 position_dependent=False,  # type: bool
                 rbf_sigma=None,  # type: float
                 random_state=None,
                 expand_factor=10,  # type: Optional[int]
                 token_pattern=None,  # type: str
                 ):
        # type: (...) -> None
        self.n_samples = n_samples
        self.random_state = random_state
        self.expand_factor = expand_factor
        self.rng_ = check_random_state(random_state)
        if clf is None:
            clf = SGDClassifier(loss='log',
                                penalty='elasticnet',
                                alpha=1e-3,
                                random_state=self.rng_)
        self.clf = clf

        if char_based is None:
            if token_pattern is None:
                self.char_based = False
                self.token_pattern = DEFAULT_TOKEN_PATTERN
            else:
                self.char_based = None
                self.token_pattern = token_pattern
        else:
            if token_pattern is not None:
                raise ValueError("Use either ``char_based`` or "
                                 "``token_pattern``, but not both.")
            self.char_based = char_based
            self.token_pattern = (CHAR_TOKEN_PATTERN if char_based
                                  else DEFAULT_TOKEN_PATTERN)

        if sampler is None:
            sampler = MaskingTextSamplers(
                sampler_params=[{'bow': False}, {'bow': True}],
                weights=[0.7, 0.3],
                token_pattern=self.token_pattern,
                random_state=self.rng_,
            )
        self.sampler = sampler
        self.rbf_sigma = rbf_sigma
        self.position_dependent = position_dependent
        if position_dependent:
            if vec is not None:
                raise ValueError("Custom vectorizers are only supported with "
                                 "position_dependent=False")
        else:
            if vec is None:
                if self.char_based:
                    vec = CountVectorizer(
                        analyzer='char',
                        ngram_range=(2, 5),
                    )
                else:
                    vec = CountVectorizer(token_pattern=self.token_pattern)
            self.vec = vec

    def fit(self,
            doc,             # type: str
            predict_proba,   # type: Callable[[Any], Any]
            ):
        # type: (...) -> TextExplainer
        self.doc_ = doc

        if self.position_dependent:
            samples, sims, mask, text = self.sampler.sample_near_with_mask(
                doc=doc,
                n_samples=self.n_samples
            )
            self.vec_ = SingleDocumentVectorizer(
                token_pattern=self.token_pattern
            ).fit([doc])
            X = ~mask
        else:
            self.vec_ = clone(self.vec).fit([doc])
            samples, sims = self.sampler.sample_near(
                doc=doc,
                n_samples=self.n_samples
            )
            X = self.vec_.transform(samples)

        if self.rbf_sigma is not None:
            sims = rbf(1-sims, sigma=self.rbf_sigma)

        self.samples_ = samples
        self.similarity_ = sims
        self.X_ = X
        self.y_proba_ = predict_proba(samples)
        self.clf_ = clone(self.clf)

        self.metrics_ = _train_local_classifier(
            estimator=self.clf_,
            samples=X,
            similarity=sims,
            y_proba=self.y_proba_,
            expand_factor=self.expand_factor,
            random_state=self.rng_
        )
        return self

    def show_prediction(self, **kwargs):
        return eli5.show_prediction(self.clf_, self.doc_, vec=self.vec_,
                                    **kwargs)

    def explain_prediction(self, **kwargs):
        return eli5.explain_prediction(self.clf_, self.doc_, vec=self.vec_,
                                       **kwargs)

    def show_weights(self, **kwargs):
        return eli5.show_weights(self.clf_, vec=self.vec_, **kwargs)

    def explain_weights(self, **kwargs):
        return eli5.explain_weights(self.clf_, vec=self.vec_, **kwargs)
