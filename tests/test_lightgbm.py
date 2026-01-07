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
from .test_sklearn_explain_prediction import (
    assert_linear_regression_explained,
    test_explain_prediction_pandas as _check_explain_prediction_pandas,
    test_explain_clf_binary_iris as _check_binary_classifier,
)
from .utils import format_as_all, check_targets_scores, get_all_features


@pytest.fixture()
def lgb_clf():
    return LGBMClassifier(
        n_estimators=10,
        min_child_samples=2,
        min_child_weight=1,
        seed=42,
    )


@pytest.mark.parametrize(['importance_type'], [['gain'], ['split'], ['weight']])
def test_explain_weights(newsgroups_train, lgb_clf, importance_type):
    _check_rf_classifier(newsgroups_train, lgb_clf, importance_type=importance_type)


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


def test_explain_prediction_clf_binary_iris(iris_train_binary):
    clf = LGBMClassifier(n_estimators=100, max_depth=2,
                         min_child_samples=1, min_child_weight=1)
    _check_binary_classifier(clf, iris_train_binary)


def test_explain_prediction_clf_multitarget(newsgroups_train):
    docs, ys, target_names = newsgroups_train
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop_words = set(ENGLISH_STOP_WORDS) | {'does', 'just'}
    vec = CountVectorizer(stop_words=stop_words, dtype=np.float64)
    xs = vec.fit_transform(docs)
    clf = LGBMClassifier(n_estimators=100, max_depth=2,
                         min_child_samples=1, min_child_weight=1,
                         random_state=42)
    clf.fit(xs, ys)
    doc = 'computer graphics in space: a new religion'
    res = explain_prediction(clf, doc, vec=vec, target_names=target_names)
    format_as_all(res, clf)
    check_targets_scores(res)
    graphics_weights = res.targets[1].feature_weights
    assert 'computer' in get_all_features(graphics_weights.pos)
    religion_weights = res.targets[3].feature_weights
    assert 'religion' in get_all_features(religion_weights.pos)

    top_target_res = explain_prediction(clf, doc, vec=vec, top_targets=2)
    assert len(top_target_res.targets) == 2
    assert sorted(t.proba for t in top_target_res.targets) == sorted(
        t.proba for t in res.targets)[-2:]


def test_explain_prediction_single_leaf_tree(iris_train):
    X, y, feature_names, target_names = iris_train
    clf = LGBMClassifier(n_estimators=100)
    clf.fit(X, y)
    # at least one of the trees has only a single leaf

    res = explain_prediction(clf, X[0], target_names=target_names)
    format_as_all(res, clf)
    check_targets_scores(res)


def test_explain_prediction_regression(boston_train):
    assert_linear_regression_explained(
        boston_train, LGBMRegressor(), explain_prediction,
        reg_has_intercept=True)


def test_explain_prediction_pandas(boston_train):
    _check_explain_prediction_pandas(LGBMRegressor(), boston_train)


def test_explain_weights_feature_names_pandas(boston_train):
    pd = pytest.importorskip('pandas')
    X, y, feature_names = boston_train
    df = pd.DataFrame(X, columns=feature_names)
    reg = LGBMRegressor().fit(df, y)

    # it shoud pick up feature names from DataFrame columns
    res = explain_weights(reg)
    for expl in format_as_all(res, reg):
        assert 'PTRATIO' in expl

    # it is possible to override DataFrame feature names
    numeric_feature_names = ["zz%s" % idx for idx in range(len(feature_names))]
    res = explain_weights(reg, feature_names=numeric_feature_names)
    for expl in format_as_all(res, reg):
        assert 'zz12' in expl
