# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier

from eli5._feature_names import FeatureNames
from eli5.sklearn.utils import (
    get_feature_names,
    get_default_target_names,
    has_intercept,
    is_multiclass_classifier,
    is_multitarget_regressor,
    get_num_features,
)
from .utils import SGD_KWARGS


@pytest.mark.parametrize(['clf', 'intercept'], [
    [SGDClassifier(**SGD_KWARGS), True],
    [SGDClassifier(fit_intercept=False, **SGD_KWARGS), False],
    [OneVsRestClassifier(SGDClassifier(**SGD_KWARGS)), True],
    [OneVsRestClassifier(
        SGDClassifier(fit_intercept=False, **SGD_KWARGS)), False],
])
def test_has_intercept(newsgroups_train, clf, intercept):
    vec = TfidfVectorizer()
    X = vec.fit_transform(newsgroups_train[0])
    clf.fit(X, newsgroups_train[1])
    assert has_intercept(clf) == intercept


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

    def _names(*args, **kwargs):
        return set(get_feature_names(*args, **kwargs))

    for y in [[0, 1, 2], [0, 1, 0]]:  # multiclass, binary
        vec = CountVectorizer()
        X = vec.fit_transform(docs)

        clf = LogisticRegression()
        clf.fit(X, y)

        fnames = get_feature_names(clf, vec)
        assert isinstance(fnames, FeatureNames)
        assert repr(fnames) == '<FeatureNames: 2 features with bias>'
        assert _names(clf, vec) == {'hello', 'world', '<BIAS>'}
        assert _names(clf, vec, 'B') == {'hello', 'world', 'B'}
        assert _names(clf) == {'x0', 'x1', '<BIAS>'}
        assert _names(clf, feature_names=['a', 'b']) == {'a', 'b', '<BIAS>'}
        assert _names(clf, feature_names=['a', 'b'],
                                   bias_name='bias') == {'a', 'b', 'bias'}
        assert _names(clf, feature_names=np.array(['a', 'b'])) == {'a', 'b', '<BIAS>'}
        assert _names(clf, feature_names=FeatureNames(['a', 'b'])) == {'a', 'b', '<BIAS>'}
        assert _names(clf, feature_names=FeatureNames(
            n_features=2, unkn_template='F%d')) == {'F0', 'F1', '<BIAS>'}

        with pytest.raises(ValueError):
            get_feature_names(clf, feature_names=['a'])

        with pytest.raises(ValueError):
            get_feature_names(clf, feature_names=['a', 'b', 'c'])

        with pytest.raises(ValueError):
            get_feature_names(clf, feature_names=FeatureNames(['a', 'b', 'c']))

        clf2 = LogisticRegression(fit_intercept=False)
        clf2.fit(X, y)
        assert _names(clf2, vec) == {'hello', 'world'}
        assert _names(clf2, feature_names=['hello', 'world']) == {'hello', 'world'}


def test_get_feature_names_1dim_coef():
    clf = SGDRegressor(fit_intercept=False, **SGD_KWARGS)
    X, y = make_regression(n_targets=1, n_features=3)
    clf.fit(X, y)
    assert set(get_feature_names(clf)) == {'x0', 'x1', 'x2'}


def test_get_default_target_names():
    clf = SGDRegressor(**SGD_KWARGS)
    X, y = make_regression(n_targets=1, n_features=3)
    clf.fit(X, y)
    assert set(get_default_target_names(clf)) == {'y'}

    clf = ElasticNet()
    X, y = make_regression(n_targets=2, n_features=3)
    clf.fit(X, y)
    assert set(get_default_target_names(clf)) == {'y0', 'y1'}


@pytest.mark.parametrize(['clf'], [
    [LogisticRegression()],
    [RandomForestClassifier()],
    [GaussianNB()],
    [DecisionTreeClassifier()],
    [OneVsRestClassifier(DecisionTreeClassifier())],
    [OneVsRestClassifier(LogisticRegression())],
    [BernoulliNB()],
])
def test_get_num_features(clf):
    X_bin, y_bin = make_classification(n_features=20, n_classes=2)
    X, y = make_classification(n_features=20, n_informative=4, n_classes=3)

    clf.fit(X, y)
    assert get_num_features(clf) == 20

    clf.fit(X_bin, y_bin)
    assert get_num_features(clf) == 20
