# -*- coding: utf-8 -*-
from singledispatch import singledispatch

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC

from eli5.sklearn.utils import (
    get_feature_names,
    get_coef,
    is_multiclass_classifier,
    is_probabilistic_classifier,
    has_intercept,
    rename_label,
)
from eli5._feature_weights import get_top_features_dict


_TOP = 20


@singledispatch
def explain_prediction(clf, doc, vec=None, top=_TOP, target_names=None,
                       feature_names=None):
    """ Return an explanation of an estimator """
    return {
        "estimator": repr(clf),
        "description": "Error: estimator %r is not supported" % clf,
    }


@explain_prediction.register(LogisticRegression)
@explain_prediction.register(LogisticRegressionCV)
@explain_prediction.register(SGDClassifier)
@explain_prediction.register(PassiveAggressiveClassifier)
@explain_prediction.register(Perceptron)
@explain_prediction.register(LinearSVC)
def explain_prediction_linear(
        clf, doc, vec=None, top=_TOP, target_names=None,
        feature_names=None, vectorized=False):
    """ Explain prediction of a linear classifier. """
    feature_names = get_feature_names(clf, vec, feature_names=feature_names)
    X = _get_X(doc, vec=vec, vectorized=vectorized)

    if is_probabilistic_classifier(clf):
        proba = clf.predict_proba(X)[0]
    else:
        proba = None
    score = clf.decision_function(X)[0]

    if has_intercept(clf):
        X = _add_intercept(X)
    x = X[0]

    res = {
        "estimator": repr(clf),
        "method": "linear model",
        "classes": [],
    }

    def _weights(label_id):
        coef = get_coef(clf, label_id)
        scores = _multiply(x, coef)
        return get_top_features_dict(feature_names, scores, top)

    def _label(label_id, label):
        return rename_label(label_id, label, target_names)

    if is_multiclass_classifier(clf):
        for label_id, label in enumerate(clf.classes_):
            class_info = {
                'class': _label(label_id, label),
                'feature_weights': _weights(label_id),
                'score': score[label_id],
            }
            if proba is not None:
                class_info['proba'] = proba[label_id]
            res['classes'].append(class_info)
        return res
    else:
        class_info = {
            'class': _label(1, clf.classes_[1]),
            'feature_weights': _weights(0),
            'score': score,
        }
        if proba is not None:
            class_info['proba'] = proba[1]
        res['classes'].append(class_info)

    return res


def _multiply(X, coef):
    if sp.issparse(X):
        return X.multiply(sp.csr_matrix(coef))
    else:
        return np.multiply(X, coef)


def _add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    if sp.issparse(X):
        return sp.hstack([X, intercept]).tocsr()
    else:
        return np.hstack([X, intercept])


def _get_X(doc, vec=None, vectorized=False):
    if vec is None or vectorized:
        X = np.array([doc]) if isinstance(doc, np.ndarray) else doc
    else:
        X = vec.transform([doc])
    if sp.issparse(X):
        X = X.toarray()
    return X
