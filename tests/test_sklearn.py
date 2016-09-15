# -*- coding: utf-8 -*-
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    SGDClassifier,
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

from eli5.sklearn import (
    get_feature_names,
    get_coef,
    has_intercept,
    is_multiclass_classifier
)
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

    def _top(class_name):
        for expl in res['classes']:
            if expl['class'] != class_name:
                continue
            pos = {name for name, value in expl['feature_weights']['pos']}
            neg = {name for name, value in expl['feature_weights']['neg']}
            return pos, neg

    assert [cl['class'] for cl in res['classes']] == class_names

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

        clf2 = LogisticRegression(fit_intercept=False)
        clf2.fit(X, y)
        assert set(get_feature_names(clf2, vec)) == {'hello', 'world'}


def test_unsupported():
    vec = CountVectorizer()
    clf = BaseEstimator()
    res = explain_weights(clf, vec)
    assert 'Error' in res['description']
