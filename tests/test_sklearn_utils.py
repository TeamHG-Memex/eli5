# -*- coding: utf-8 -*-
from __future__ import absolute_import

import pytest
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

from eli5.sklearn.utils import (
    get_feature_names,
    get_coef,
    has_intercept,
    is_multiclass_classifier
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
