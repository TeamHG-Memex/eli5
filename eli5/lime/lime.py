# -*- coding: utf-8 -*-
"""
An impementation of LIME (http://arxiv.org/abs/1602.04938), an algorithm to
explain predictions of black-box models.

Input:

1. a black-box classifier (or regressor - not implemented here).
2. an example to explain.

Main idea:

1. Generate a fake dataset from the example to explain: generated instances
   are changed versions of the example (e.g. for text it could be the same text,
   but with some words removed); generated labels are black-box classifier
   predictions for these generated examples.

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

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

from eli5.lime import textutils
from eli5.utils import vstack


def get_local_classifier(doc,
                         predict_proba,
                         local_clf,
                         local_vec,
                         generate_perturbations,
                         n_samples=200,
                         expand_factor=10,
                         test_size=0.3,
                         ):
    docs, similarities = generate_perturbations(doc, n_samples=n_samples)
    y_proba = predict_proba(docs)
    y_best = y_proba.argmax(axis=1)

    (docs_train, docs_test,
     y_proba_train, y_proba_test,
     y_best_train, y_best_test) = train_test_split(docs, y_proba, y_best,
                                                   test_size=test_size)

    # scikit-learn can't optimize cross-entropy directly if target
    # probability values are not indicator vectors. Probability information
    # is helpful because it could be hard to get enough examples of all classes
    # automatically. As a workaround here we're expanding the dataset
    # according to target probabilities. Use expand_factor=None to turn
    # it off (e.g. if probability scores are 0/1 in a first place).
    #
    # XXX: in the original lime code instead of a probabilitsic classifier
    # they build several regression models which try to output probabilities.
    #
    # XXX: similarities are currently unused; using sample_weights
    # doesn't seem to improve quality. TODO: investigate it.
    X_train = local_vec.fit_transform(docs_train)

    if expand_factor:
        X_train, y_train = zip(*expand_dataset(X_train, y_proba_train,
                                               expand_factor))
        X_train = vstack(X_train)
    else:
        y_train = y_proba_train.argmax(axis=1)

    # TODO: feature selection
    local_clf.fit(X_train, y_train)

    X_test = local_vec.transform(docs_test)
    score = local_clf.score(X_test, y_best_test)
    return local_clf, local_vec, score


def get_local_classifier_text(text, predict_proba, n_samples=1000,
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

    return get_local_classifier(
        doc=text,
        predict_proba=predict_proba,
        local_clf=clf,
        local_vec=vec,
        generate_perturbations=textutils.generate_perturbations,
        n_samples=n_samples,
        expand_factor=expand_factor,
    )


def expand_dataset(X, y_proba, factor=10):
    """
    Convert a dataset with float multiclass probabilities to a dataset
    with indicator probabilities by duplicating X rows and sampling
    true labels.
    """
    n_classes = y_proba.shape[1]
    classes = np.arange(n_classes, dtype=int)
    for x, probs in zip(X, y_proba):
        for label in np.random.choice(classes, size=factor, p=probs):
            yield x, label
