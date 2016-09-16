# -*- coding: utf-8 -*-
from __future__ import absolute_import
from pprint import pprint

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
    PassiveAggressiveClassifier,
    Perceptron,
)
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator
import pytest

from eli5.sklearn import explain_prediction
from eli5.formatters import format_as_text


@pytest.mark.parametrize(['clf'], [
    [LogisticRegression()],
    [LogisticRegression(multi_class='multinomial', solver='lbfgs')],
    [LogisticRegression(fit_intercept=False)],
    [LogisticRegressionCV()],
    [SGDClassifier(random_state=42)],
    [SGDClassifier(loss='log', random_state=42)],
    [PassiveAggressiveClassifier()],
    [Perceptron()],
    [LinearSVC()],
])
def test_explain_linear(newsgroups_train, clf):
    docs, y, class_names = newsgroups_train
    vec = TfidfVectorizer()

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    res = explain_prediction(clf, vec, docs[0], class_names=class_names, top=20)
    expl = format_as_text(res)
    print(expl)
    pprint(res)

    for e in res['classes']:
        if e['class'] != 'comp.graphics':
            continue
        pos = {name for name, value in e['feature_weights']['pos']}
        assert 'file' in pos

    for label in class_names:
        assert str(label) in expl
    assert 'file' in expl


def test_explain_linear_binary(newsgroups_train_binary):
    docs, y, class_names = newsgroups_train_binary
    vec = TfidfVectorizer()
    clf = LogisticRegression()
    X = vec.fit_transform(docs)
    clf.fit(X, y)

    res = explain_prediction(clf, vec, docs[0], class_names=class_names, top=20)
    expl = format_as_text(res)
    print(expl)
    pprint(res)

    assert len(res['classes']) == 1
    e = res['classes'][0]
    assert e['class'] == 'comp.graphics'
    neg = {name for name, value in e['feature_weights']['neg']}
    assert 'freedom' in neg
    assert 'comp.graphics' in expl
    assert 'freedom' in expl


def test_unsupported():
    vec = CountVectorizer()
    clf = BaseEstimator()
    res = explain_prediction(clf, vec, 'hello, world')
    assert 'Error' in res['description']
