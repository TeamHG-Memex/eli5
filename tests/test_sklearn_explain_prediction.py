# -*- coding: utf-8 -*-
from __future__ import absolute_import
from pprint import pprint

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
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

    res = explain_prediction(clf, docs[0], vec=vec, class_names=class_names, top=20)
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

    res = explain_prediction(clf, docs[0], vec, class_names=class_names, top=20)
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

    res_vectorized = explain_prediction(
        clf, vec.transform([docs[0]])[0], vec, class_names=class_names,
        top=20, vectorized=True)
    assert res_vectorized == res


def test_explain_linear_dense():
    clf = LogisticRegression()
    data = [{'day': 'mon', 'moon': 'full'},
            {'day': 'tue', 'moon': 'rising'},
            {'day': 'tue', 'moon': 'rising'},
            {'day': 'mon', 'moon': 'rising'}]
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(data)
    clf.fit(X, [0, 1, 1, 0])
    test_day = {'day': 'tue', 'moon': 'full'}
    class_names = ['sunny', 'shady']
    res1 = explain_prediction(clf, test_day, vec, class_names=class_names)
    expl1 = format_as_text(res1)
    print(expl1)
    assert 'day=tue' in expl1
    [test_day_vec] = vec.transform(test_day)
    res2 = explain_prediction(
        clf, test_day_vec, class_names=class_names,
        vectorized=True, feature_names=vec.get_feature_names())
    expl2 = format_as_text(res1)
    print(expl2)
    assert res1 == res2


def test_unsupported():
    vec = CountVectorizer()
    clf = BaseEstimator()
    res = explain_prediction(clf, 'hello, world', vec)
    assert 'Error' in res['description']


def test_without_vec():
    clf = LogisticRegression()
    clf.fit(np.array([[1], [0]]), np.array([0, 1]))
    with pytest.raises(ValueError) as excinfo:
        explain_prediction(clf, 'hello, world')
    assert 'vec' in str(excinfo.value)
    assert 'vectorized' in str(excinfo.value)
