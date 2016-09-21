# -*- coding: utf-8 -*-
from __future__ import absolute_import

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, ElasticNet, SGDRegressor

from eli5.sklearn.utils import (
    get_feature_names,
    get_target_names,
    has_intercept,
    is_multiclass_classifier,
    is_multitarget_regressor,
)


def test_has_intercept(newsgroups_train):
    vec = TfidfVectorizer()
    X = vec.fit_transform(newsgroups_train[0])
    clf = LogisticRegression()
    clf.fit(X, newsgroups_train[1])
    assert has_intercept(clf)

    clf2 = LogisticRegression(fit_intercept=False)
    clf2.fit(X, newsgroups_train[1])
    assert not has_intercept(clf2)


def test_is_multiclass():
    X, y = make_classification(n_classes=2)
    clf = LogisticRegression()
    clf.fit(X, y)
    assert not is_multiclass_classifier(clf)

    X, y = make_classification(n_classes=3, n_informative=5)
    clf2 = LogisticRegression()
    clf2.fit(X, y)
    assert is_multiclass_classifier(clf2)


def test_is_multitarget_regressor():
    X, y = make_regression(n_targets=1)
    clf = ElasticNet()
    clf.fit(X, y)
    assert not is_multitarget_regressor(clf)

    X, y = make_regression(n_targets=2)
    clf = ElasticNet()
    clf.fit(X, y)
    assert is_multitarget_regressor(clf)


def test_get_feature_names():
    docs = ['hello world', 'hello', 'world']

    for y in [[0, 1, 2], [0, 1, 0]]:  # multiclass, binary
        vec = CountVectorizer()
        X = vec.fit_transform(docs)

        clf = LogisticRegression()
        clf.fit(X, y)

        assert set(get_feature_names(clf, vec)) == {'hello', 'world', '<BIAS>'}
        assert set(get_feature_names(clf, vec, 'B')) == {'hello', 'world', 'B'}
        assert set(get_feature_names(clf)) == {'x0', 'x1', '<BIAS>'}
        assert set(get_feature_names(clf, feature_names=['a', 'b'])) == {'a', 'b', '<BIAS>'}
        assert set(get_feature_names(clf, feature_names=['a', 'b'],
                                     bias_name='bias')) == {'a', 'b', 'bias'}

        with pytest.raises(ValueError):
            get_feature_names(clf, feature_names=['a'])

        with pytest.raises(ValueError):
            get_feature_names(clf, feature_names=['a', 'b', 'c'])

        clf2 = LogisticRegression(fit_intercept=False)
        clf2.fit(X, y)
        assert set(get_feature_names(clf2, vec)) == {'hello', 'world'}
        assert set(get_feature_names(
            clf2, feature_names=['hello', 'world'])) == {'hello', 'world'}


def test_get_feature_names_1dim_coef():
    clf = SGDRegressor(fit_intercept=False)
    X, y = make_regression(n_targets=1, n_features=3)
    clf.fit(X, y)
    assert set(get_feature_names(clf)) == {'x0', 'x1', 'x2'}


def test_get_target_names():
    clf = SGDRegressor()
    X, y = make_regression(n_targets=1, n_features=3)
    clf.fit(X, y)
    assert set(get_target_names(clf)) == {'y'}

    clf = ElasticNet()
    X, y = make_regression(n_targets=2, n_features=3)
    clf.fit(X, y)
    assert set(get_target_names(clf)) == {'y0', 'y1'}
