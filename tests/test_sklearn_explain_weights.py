# -*- coding: utf-8 -*-
from __future__ import absolute_import
from functools import partial

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
    SGDRegressor,
    PassiveAggressiveClassifier,
    Perceptron,
)
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
import pytest

from eli5.sklearn import explain_weights
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

    res = explain_weights(clf, vec, class_names=class_names, top=20)
    expl = format_as_text(res)
    print(expl)

    assert [cl['class'] for cl in res['classes']] == class_names

    _top = partial(top_pos_neg, res['classes'], 'class')
    pos, neg = _top('sci.space')
    assert 'space' in pos

    pos, neg = _top('alt.atheism')
    assert 'atheists' in pos

    pos, neg = _top('talk.religion.misc')
    assert 'jesus' in pos

    assert 'space' in expl
    assert 'atheists' in expl
    for label in class_names:
        assert str(label) in expl


def top_pos_neg(classes, key, class_name):
    for expl in classes:
        if expl[key] != class_name:
            continue
        pos = {name for name, value in expl['feature_weights']['pos']}
        neg = {name for name, value in expl['feature_weights']['neg']}
        return pos, neg


def test_explain_linear_tuple_top(newsgroups_train):
    docs, y, class_names = newsgroups_train
    vec = TfidfVectorizer()
    clf = LogisticRegression()

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    res_neg = explain_weights(clf, vec, class_names=class_names, top=(0, 10))
    expl_neg = format_as_text(res_neg)
    print(expl_neg)

    for cl in res_neg['classes']:
        assert len(cl['feature_weights']['pos']) == 0
        assert len(cl['feature_weights']['neg']) == 10

    assert "+0." not in expl_neg

    res_pos = explain_weights(clf, vec, class_names=class_names, top=(10, 2))
    expl_pos = format_as_text(res_pos)
    print(expl_pos)

    for cl in res_pos['classes']:
        assert len(cl['feature_weights']['pos']) == 10
        assert len(cl['feature_weights']['neg']) == 2


@pytest.mark.parametrize(['clf'], [
    [RandomForestClassifier(n_estimators=100)],
    [ExtraTreesClassifier(n_estimators=100)],
    [GradientBoostingClassifier()],
    [AdaBoostClassifier(learning_rate=0.1, n_estimators=200)],
    [DecisionTreeClassifier()],
])
def test_explain_random_forest(newsgroups_train, clf):
    docs, y, class_names = newsgroups_train
    vec = TfidfVectorizer()
    X = vec.fit_transform(docs)
    clf.fit(X.toarray(), y)

    res = explain_weights(clf, vec, class_names=class_names, top=30)
    expl = format_as_text(res)
    print(expl)
    assert 'feature importances' in expl
    assert 'that 0.' in expl  # high-ranked feature


def test_explain_empty(newsgroups_train):
    clf = LogisticRegression(C=0.01, penalty='l1')
    docs, y, class_names = newsgroups_train
    vec = TfidfVectorizer()

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    res = explain_weights(clf, vec, class_names=class_names, top=20)
    expl = format_as_text(res)
    print(expl)

    assert [cl['class'] for cl in res['classes']] == class_names


def test_unsupported():
    vec = CountVectorizer()
    clf = BaseEstimator()
    res = explain_weights(clf, vec)
    assert 'Error' in res['description']


@pytest.mark.parametrize(['clf'], [
    [SGDRegressor(random_state=13)],
])
def test_explain_linear_regression(boston_train, clf):
    X, y, feature_names = boston_train
    clf.fit(X, y)
    res = explain_weights(clf)
    expl = format_as_text(res)
    print(expl)

    assert 'x12' in expl
    assert 'x9' in expl
    assert '<BIAS>' in expl

    pos, neg = top_pos_neg(res['targets'], 'target', 'y')
    assert 'x12' in pos
    assert 'x9' in neg
    assert '<BIAS>' in neg

