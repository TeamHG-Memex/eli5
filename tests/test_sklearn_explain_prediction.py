# -*- coding: utf-8 -*-
from __future__ import absolute_import
from pprint import pprint

import pytest
from sklearn.datasets import make_regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lars,
    Lasso,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    PassiveAggressiveClassifier,
    Perceptron,
    Ridge,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.multiclass import OneVsRestClassifier

from eli5 import explain_prediction
from eli5.sklearn.utils import has_intercept
from .utils import format_as_all, strip_blanks, get_all_features


def assert_multiclass_linear_classifier_explained(newsgroups_train, clf,
                                                  explain_prediction):
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    get_res = lambda: explain_prediction(
        clf, docs[0], vec=vec, target_names=target_names, top=20)
    res = get_res()
    pprint(res)
    expl_text, expl_html = format_as_all(res, clf)

    for e in res.targets:
        if e.target != 'comp.graphics':
            continue
        pos = get_all_features(e.feature_weights.pos)
        assert 'file' in pos

    for expl in [expl_text, expl_html]:
        for label in target_names:
            assert str(label) in expl
        assert 'file' in expl

    assert res == get_res()


def assert_linear_regression_explained(boston_train, reg, explain_prediction):
    X, y, feature_names = boston_train
    reg.fit(X, y)
    res = explain_prediction(reg, X[0])
    expl_text, expl_html = format_as_all(res, reg)

    assert len(res.targets) == 1
    target = res.targets[0]
    assert target.target == 'y'
    pos, neg = (get_all_features(target.feature_weights.pos),
                get_all_features(target.feature_weights.neg))
    assert 'x11' in pos or 'x11' in neg

    if has_intercept(reg):
        assert '<BIAS>' in pos or '<BIAS>' in neg
        assert '<BIAS>' in expl_text
        assert '&lt;BIAS&gt;' in expl_html
    else:
        assert '<BIAS>' not in pos and '<BIAS>' not in neg
        assert '<BIAS>' not in expl_text
        assert 'BIAS' not in expl_html

    for expl in [expl_text, expl_html]:
        assert 'x11' in expl
        assert '(score' in expl
    assert "'y'" in expl_text
    assert '<b>y</b>' in strip_blanks(expl_html)

    assert res == explain_prediction(reg, X[0])


def assert_multitarget_linear_regression_explained(reg, explain_prediction):
    X, y = make_regression(n_samples=100, n_targets=3, n_features=10,
                           random_state=42)
    reg.fit(X, y)
    res = explain_prediction(reg, X[0])
    expl_text, expl_html = format_as_all(res, reg)

    assert len(res.targets) == 3
    target = res.targets[1]
    assert target.target == 'y1'
    pos, neg = (get_all_features(target.feature_weights.pos),
                get_all_features(target.feature_weights.neg))
    assert 'x8' in pos or 'x8' in neg
    assert '<BIAS>' in pos or '<BIAS>' in neg

    assert 'x8' in expl_text
    assert '<BIAS>' in expl_text
    assert "'y2'" in expl_text

    assert res == explain_prediction(reg, X[0])


@pytest.mark.parametrize(['clf'], [
    [LogisticRegression(random_state=42)],
    [LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs')],
    [LogisticRegression(random_state=42, fit_intercept=False)],
    [LogisticRegressionCV(random_state=42)],
    [SGDClassifier(random_state=42)],
    [SGDClassifier(loss='log', random_state=42)],
    [PassiveAggressiveClassifier(random_state=42)],
    [Perceptron(random_state=42)],
    [LinearSVC(random_state=42)],
    [OneVsRestClassifier(LogisticRegression(random_state=42))],
])
def test_explain_linear(newsgroups_train, clf):
    assert_multiclass_linear_classifier_explained(newsgroups_train, clf,
                                                  explain_prediction)


@pytest.mark.parametrize(['reg'], [
    [ElasticNet(random_state=42)],
    [ElasticNetCV(random_state=42)],
    [Lars()],
    [Lasso(random_state=42)],
    [LinearRegression()],
    [LinearSVR(random_state=42)],
    [Ridge(random_state=42)],
    [RidgeCV()],
    [SGDRegressor(random_state=42)],
])
def test_explain_linear_regression(boston_train, reg):
    assert_linear_regression_explained(boston_train, reg, explain_prediction)


@pytest.mark.parametrize(['reg'], [
    [ElasticNet(random_state=42)],
    [Lars()],
    [Lasso(random_state=42)],
    [LinearRegression()],
    [Ridge(random_state=42)],
])
def test_explain_linear_regression_multitarget(reg):
    assert_multitarget_linear_regression_explained(reg, explain_prediction)
