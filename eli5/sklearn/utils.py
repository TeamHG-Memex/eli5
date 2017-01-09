# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import Any, Optional

import numpy as np  # type: ignore
import scipy.sparse as sp  # type: ignore
from sklearn.multiclass import OneVsRestClassifier  # type: ignore
from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore

from eli5.sklearn.unhashing import InvertableHashingVectorizer, is_invhashing
from eli5._feature_names import FeatureNames


def is_multiclass_classifier(clf):
    # type: (Any) -> bool
    """
    Return True if a classifier is multiclass or False if it is binary.
    """
    return clf.coef_.shape[0] > 1


def is_multitarget_regressor(clf):
    # type: (Any) -> bool
    """
    Return True if a regressor is multitarget
    or False if it predicts a single target.
    """
    return len(clf.coef_.shape) > 1 and clf.coef_.shape[0] > 1


def is_probabilistic_classifier(clf):
    # type: (Any) -> bool
    """ Return True if a classifier can return probabilities """
    if not hasattr(clf, 'predict_proba'):
        return False
    if isinstance(clf, OneVsRestClassifier):
        # It currently has a predict_proba method, but does not check if
        # wrapped estimator has a predict_proba method.
        return hasattr(clf.estimator, 'predict_proba')
    return True


def predict_proba(clf, X):
    # type: (Any, Any) -> Optional[np.ndarray]
    """ Return result of predict_proba, if the classifier supports it, or None.
    """
    if is_probabilistic_classifier(clf):
        try:
            proba, = clf.predict_proba(X)
            return proba
        except NotImplementedError:
            return None
    else:
        return None


def has_intercept(estimator):
    # type: (Any) -> bool
    """ Return True if an estimator has intercept fit. """
    if hasattr(estimator, 'fit_intercept'):
        return estimator.fit_intercept
    if hasattr(estimator, 'intercept_'):
        if estimator.intercept_ is None:
            return False
        # scikit-learn sets intercept to zero vector if it is not fit
        return np.any(estimator.intercept_)
    return False


def get_feature_names(clf, vec=None, bias_name='<BIAS>', feature_names=None,
                      num_features=None):
    # type: (Any, Any, str, Any, int) -> FeatureNames
    """
    Return a FeatureNames instance that holds all feature names
    and a bias feature.
    If vec is None or doesn't have get_feature_names() method,
    features are named x0, x1, x2, etc.
    """
    if not has_intercept(clf):
        bias_name = None

    if feature_names is None:
        if vec and hasattr(vec, 'get_feature_names'):
            return FeatureNames(vec.get_feature_names(), bias_name=bias_name)
        else:
            num_features = num_features or get_num_features(clf)
            return FeatureNames(
                n_features=num_features, unkn_template='x%d', bias_name=bias_name)

    num_features = num_features or get_num_features(clf)
    if isinstance(feature_names, FeatureNames):
        if feature_names.n_features != num_features:
            raise ValueError("feature_names has a wrong n_features: "
                             "expected=%d, got=%d" % (num_features,
                                                      feature_names.n_features))
        # Make a shallow copy setting proper bias_name
        return FeatureNames(
            feature_names.feature_names,
            n_features=num_features,
            bias_name=bias_name,
            unkn_template=feature_names.unkn_template)
    else:
        if len(feature_names) != num_features:
            raise ValueError("feature_names has a wrong length: "
                             "expected=%d, got=%d" % (num_features,
                                                      len(feature_names)))
        return FeatureNames(feature_names, bias_name=bias_name)


def get_default_target_names(estimator, num_targets=None):
    """
    Return a vector of target names: "y" if there is only one target,
    and "y0", "y1", ... if there are multiple targets.
    """
    if num_targets is None:
        if len(estimator.coef_.shape) == 1:
            num_targets = 1
        else:
            num_targets, _ = estimator.coef_.shape
    if num_targets == 1:
        target_names = ['y']
    else:
        target_names = ['y%d' % i for i in range(num_targets)]
    return np.array(target_names)


def get_coef(clf, label_id, scale=None):
    """
    Return a vector of coefficients for a given label,
    including bias feature.

    ``scale`` (optional) is a scaling vector; coef_[i] => coef[i] * scale[i] if
    scale[i] is not nan. Intercept is not scaled.
    """
    if len(clf.coef_.shape) == 2:
        # Most classifiers (even in binary case) and regressors
        coef = clf.coef_[label_id]
    elif len(clf.coef_.shape) == 1:
        # SGDRegressor stores coefficients in a 1D array
        if label_id != 0:
            raise ValueError(
                'Unexpected label_id %s for 1D coefficient' % label_id)
        coef = clf.coef_
    else:
        raise ValueError('Unexpected clf.coef_ shape: %s' % clf.coef_.shape)

    if scale is not None:
        if coef.shape != scale.shape:
            raise ValueError("scale shape is incorrect: expected %s, got %s" % (
                coef.shape, scale.shape,
            ))
        # print("shape is ok")
        not_nan = ~np.isnan(scale)
        coef = coef.copy()
        coef[not_nan] *= scale[not_nan]

    if not has_intercept(clf):
        return coef
    if label_id == 0 and not isinstance(clf.intercept_, np.ndarray):
        bias = clf.intercept_
    else:
        bias = clf.intercept_[label_id]
    return np.hstack([coef, bias])


def get_num_features(clf):
    """ Return size of a feature vector classifier expects as an input. """
    if hasattr(clf, 'coef_'):  # linear models
        return clf.coef_.shape[-1]
    elif hasattr(clf, 'feature_importances_'):  # ensembles
        return clf.feature_importances_.shape[-1]
    elif hasattr(clf, 'feature_count_'):  # naive bayes
        return clf.feature_count_.shape[-1]
    elif hasattr(clf, 'theta_'):
        return clf.theta_.shape[-1]
    elif hasattr(clf, 'estimators_') and len(clf.estimators_):  # OvR
        return get_num_features(clf.estimators_[0])
    else:
        raise ValueError("Can't figure out feature vector size for %s" % clf)


def get_X(doc, vec=None, vectorized=False, to_dense=False):
    if vec is None or vectorized:
        X = np.array([doc]) if isinstance(doc, np.ndarray) else doc
    else:
        X = vec.transform([doc])
    if to_dense and sp.issparse(X):
        X = X.toarray()
    return X


def handle_vec(clf, doc, vec, vectorized, feature_names, num_features=None):
    if isinstance(vec, HashingVectorizer) and not vectorized:
        vec = InvertableHashingVectorizer(vec)
        vec.fit([doc])
    if is_invhashing(vec) and feature_names is None:
        # Explaining predictions does not need coef_scale,
        # because it is handled by the vectorizer.
        feature_names = vec.get_feature_names(always_signed=False)
    feature_names = get_feature_names(
        clf, vec, feature_names=feature_names, num_features=num_features)
    return vec, feature_names


def add_intercept(X):
    """ Add intercept column to X """
    intercept = np.ones((X.shape[0], 1))
    if sp.issparse(X):
        return sp.hstack([X, intercept]).tocsr()
    else:
        return np.hstack([X, intercept])
