# -*- coding: utf-8 -*-
from __future__ import absolute_import
from functools import partial

from sklearn.datasets import make_regression
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    HashingVectorizer
)
from sklearn.linear_model import (
    ElasticNet,
    Lars,
    Lasso,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    PassiveAggressiveClassifier,
    Perceptron,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
import pytest

from eli5.sklearn import explain_weights, InvertableHashingVectorizer
from eli5.formatters import format_as_text


def check_newsgroups_explanation_linear(clf, vec, target_names):
    get_res = lambda: explain_weights(
        clf, vec, target_names=target_names, top=20)
    res = get_res()
    expl = format_as_text(res)
    print(expl)

    assert [cl['class'] for cl in res['classes']] == target_names

    _top = partial(top_pos_neg, res['classes'], 'class')
    pos, neg = _top('sci.space')
    assert 'space' in pos

    pos, neg = _top('alt.atheism')
    assert 'atheists' in pos

    pos, neg = _top('talk.religion.misc')
    assert 'jesus' in pos

    assert 'space' in expl
    assert 'atheists' in expl
    for label in target_names:
        assert str(label) in expl

    assert res == get_res()


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
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    check_newsgroups_explanation_linear(clf, vec, target_names)


@pytest.mark.parametrize(['clf'], [
    [LogisticRegression()],
    [LogisticRegression(fit_intercept=False)],
    [SGDClassifier(random_state=42)],
    [LinearSVC()],
])
def test_explain_linear_hashed(newsgroups_train, clf):
    docs, y, target_names = newsgroups_train
    vec = HashingVectorizer()
    ivec = InvertableHashingVectorizer(vec)

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    # use half of the docs to find common terms, to make it more realistic
    ivec.fit(docs[::2])

    check_newsgroups_explanation_linear(clf, ivec, target_names)


@pytest.mark.parametrize(['pass_feature_weights'], [[False], [True]])
def test_explain_linear_hashed_pos_neg(newsgroups_train, pass_feature_weights):
    docs, y, target_names = newsgroups_train
    # make it binary
    y = y.copy()
    y[y != 0] = 1
    target_names = [target_names[0], 'other']
    vec = HashingVectorizer(norm=None)
    ivec = InvertableHashingVectorizer(vec)

    clf = LogisticRegression(random_state=42)
    clf.fit(vec.fit_transform(docs), y)
    ivec.fit(docs)
    if pass_feature_weights:
        res = explain_weights(
            clf, top=(10, 10), target_names=target_names,
            feature_names=ivec.get_feature_names(always_signed=False),
            coef_scale=ivec.column_signs_)
    else:
        res = explain_weights(
            clf, ivec, top=(10, 10), target_names=target_names)

    # HashingVectorizer with norm=None is "the same" as CountVectorizer,
    # so we can compare it and check that explanation is almost the same.
    count_vec = CountVectorizer()
    count_clf = LogisticRegression(random_state=42)
    count_clf.fit(count_vec.fit_transform(docs), y)
    count_res = explain_weights(
        count_clf, count_vec, top=(10, 10), target_names=target_names)

    for key in ['pos', 'neg']:
        values, count_values = [
            sorted(r['classes'][0]['feature_weights'][key])
            for r in [res, count_res]]
        assert len(values) == len(count_values)
        for (name, coef), (count_name, count_coef) in zip(values, count_values):
            assert name == count_name
            assert abs(coef - count_coef) < 0.05


def top_pos_neg(classes, key, class_name):
    for expl in classes:
        if expl[key] != class_name:
            continue
        pos = {name for name, value in expl['feature_weights']['pos']}
        neg = {name for name, value in expl['feature_weights']['neg']}
        return pos, neg


def test_explain_linear_tuple_top(newsgroups_train):
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()
    clf = LogisticRegression()

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    res_neg = explain_weights(clf, vec, target_names=target_names, top=(0, 10))
    expl_neg = format_as_text(res_neg)
    print(expl_neg)

    for cl in res_neg['classes']:
        assert len(cl['feature_weights']['pos']) == 0
        assert len(cl['feature_weights']['neg']) == 10

    assert "+0." not in expl_neg

    res_pos = explain_weights(clf, vec, target_names=target_names, top=(10, 2))
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
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()
    X = vec.fit_transform(docs)
    clf.fit(X.toarray(), y)

    get_res = lambda: explain_weights(
        clf, vec, target_names=target_names, top=30)
    res = get_res()
    expl = format_as_text(res)
    print(expl)
    assert 'feature importances' in expl
    assert 'that 0.' in expl  # high-ranked feature

    assert res == get_res()


def test_explain_empty(newsgroups_train):
    clf = LogisticRegression(C=0.01, penalty='l1')
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    res = explain_weights(clf, vec, target_names=target_names, top=20)
    expl = format_as_text(res)
    print(expl)

    assert [cl['class'] for cl in res['classes']] == target_names


def test_unsupported():
    vec = CountVectorizer()
    clf = BaseEstimator()
    res = explain_weights(clf, vec)
    assert 'Error' in res['description']


@pytest.mark.parametrize(['clf'], [
    [ElasticNet(random_state=42)],
    [Lars()],
    [Lasso(random_state=42)],
    [Ridge(random_state=42)],
    [SGDRegressor(random_state=42)],
    [LinearRegression()],
    [LinearSVR()],
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
    assert 'x12' in pos or 'x12' in neg
    assert 'x9' in neg or 'x9' in pos
    assert '<BIAS>' in neg or '<BIAS>' in pos

    assert res == explain_weights(clf)


@pytest.mark.parametrize(['clf'], [
    [ElasticNet(random_state=42)],
    [Lars()],
    [Lasso(random_state=42)],
    [Ridge(random_state=42)],
    [LinearRegression()],
])
def test_explain_linear_regression_multitarget(clf):
    X, y = make_regression(n_samples=100, n_targets=3, n_features=10,
                           random_state=42)
    clf.fit(X, y)
    res = explain_weights(clf)
    expl = format_as_text(res)
    print(expl)

    assert 'x9' in expl
    assert '<BIAS>' in expl

    pos, neg = top_pos_neg(res['targets'], 'target', 'y2')
    assert 'x9' in neg or 'x9' in pos
    assert '<BIAS>' in neg or '<BIAS>' in pos

    assert res == explain_weights(clf)
