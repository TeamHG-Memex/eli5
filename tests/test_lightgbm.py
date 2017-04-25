# -*- coding: utf-8 -*-
from __future__ import absolute_import

import pytest

pytest.importorskip('lightgbm')

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from lightgbm import LGBMClassifier, LGBMRegressor

from eli5 import explain_weights, explain_prediction
from .test_sklearn_explain_weights import (
    test_explain_tree_classifier as _check_rf_classifier,
    test_explain_random_forest_and_tree_feature_filter as _check_rf_feature_filter,
    test_feature_importances_no_remaining as _check_rf_no_remaining,
    test_explain_tree_regressor as _check_rf_regressor,
)
from .test_sklearn_explain_prediction import assert_linear_regression_explained
from .utils import format_as_all, check_targets_scores, get_all_features


@pytest.fixture()
def lgb_clf():
    return LGBMClassifier(n_estimators=10,
                          min_child_samples=2,
                          min_child_weight=0)


def test_explain_weights(newsgroups_train, lgb_clf):
    _check_rf_classifier(newsgroups_train, lgb_clf)


def test_explain_weights_feature_filter(newsgroups_train, lgb_clf):
    _check_rf_feature_filter(newsgroups_train, lgb_clf)


def test_explain_weights_feature_importances_no_remaining(lgb_clf):
    _check_rf_no_remaining(lgb_clf)


def test_explain_weights_regressor(boston_train):
    reg = LGBMRegressor()
    _check_rf_regressor(reg, boston_train)


def test_explain_prediction_clf_binary(newsgroups_train_binary_big):
    docs, ys, target_names = newsgroups_train_binary_big
    vec = CountVectorizer(stop_words='english', dtype=np.float64)
    clf = LGBMClassifier(n_estimators=100, max_depth=2,
                         min_child_samples=1, min_child_weight=1)
    xs = vec.fit_transform(docs)
    clf.fit(xs, ys)
    get_res = lambda **kwargs: explain_prediction(
        clf, 'computer graphics in space: a sign of atheism',
        vec=vec, target_names=target_names, **kwargs)
    res = get_res()
    for expl in format_as_all(res, clf, show_feature_values=True):
        assert 'graphics' in expl
    check_targets_scores(res)
    weights = res.targets[0].feature_weights
    pos_features = get_all_features(weights.pos)
    neg_features = get_all_features(weights.neg)
    assert 'graphics' in pos_features
    assert 'computer' in pos_features
    assert 'atheism' in neg_features

    flt_res = get_res(feature_re='gra')
    flt_pos_features = get_all_features(flt_res.targets[0].feature_weights.pos)
    assert 'graphics' in flt_pos_features
    assert 'computer' not in flt_pos_features


def test_explain_prediction_regression(boston_train):
    assert_linear_regression_explained(
        boston_train, LGBMRegressor(), explain_prediction,
        reg_has_intercept=True)
