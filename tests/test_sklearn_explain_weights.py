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
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
import pytest

from eli5 import _graphviz
from eli5 import explain_weights
from eli5.sklearn import InvertableHashingVectorizer
from .utils import format_as_all, get_all_features, get_names_coefs


def check_newsgroups_explanation_linear(clf, vec, target_names):
    get_res = lambda: explain_weights(
        clf, vec=vec, target_names=target_names, top=20)
    res = get_res()
    expl_text, expl_html = format_as_all(res, clf)

    assert [cl.target for cl in res.targets] == target_names

    _top = partial(top_pos_neg, res)
    pos, neg = _top('sci.space')
    assert 'space' in pos

    pos, neg = _top('alt.atheism')
    assert 'atheists' in pos

    pos, neg = _top('talk.religion.misc')
    assert 'jesus' in pos or 'christians' in pos

    for expl in [expl_text, expl_html]:
        assert 'space' in expl
        assert 'atheists' in expl
        for label in target_names:
            assert str(label) in expl

    assert res == get_res()


@pytest.mark.parametrize(['clf'], [
    [LogisticRegression(random_state=42)],
    [LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs')],
    [LogisticRegression(random_state=42, fit_intercept=False)],
    [LogisticRegressionCV(random_state=42)],
    [SGDClassifier(random_state=42)],
    [SGDClassifier(random_state=42, loss='log')],
    [PassiveAggressiveClassifier(random_state=42)],
    [Perceptron(random_state=42)],
    [LinearSVC(random_state=42)],
])
def test_explain_linear(newsgroups_train, clf):
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    check_newsgroups_explanation_linear(clf, vec, target_names)


@pytest.mark.parametrize(['clf'], [
    [LogisticRegression(random_state=42)],
    [LogisticRegression(random_state=42, fit_intercept=False)],
    [SGDClassifier(random_state=42)],
    [LinearSVC(random_state=42)],
])
def test_explain_linear_hashed(newsgroups_train, clf):
    docs, y, target_names = newsgroups_train
    vec = HashingVectorizer(n_features=10000)
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
        count_clf, vec=count_vec, top=(10, 10), target_names=target_names)

    for key in ['pos', 'neg']:
        values, count_values = [
            sorted(get_names_coefs(getattr(r.targets[0].feature_weights, key)))
            for r in [res, count_res]]
        assert len(values) == len(count_values)
        for (name, coef), (count_name, count_coef) in zip(values, count_values):
            assert name == count_name
            assert abs(coef - count_coef) < 0.05


def top_pos_neg(expl, target_name):
    for target in expl.targets:
        if target.target == target_name:
            pos = get_all_features(target.feature_weights.pos)
            neg = get_all_features(target.feature_weights.neg)
            return pos, neg


def test_explain_linear_tuple_top(newsgroups_train):
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()
    clf = LogisticRegression(random_state=42)

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    res_neg = explain_weights(clf, vec=vec, target_names=target_names, top=(0, 10))
    expl_neg, _ = format_as_all(res_neg, clf)

    for target in res_neg.targets:
        assert len(target.feature_weights.pos) == 0
        assert len(target.feature_weights.neg) == 10

    assert "+0." not in expl_neg

    res_pos = explain_weights(clf, vec=vec, target_names=target_names, top=(10, 2))
    format_as_all(res_pos, clf)

    for target in res_pos.targets:
        assert len(target.feature_weights.pos) == 10
        assert len(target.feature_weights.neg) == 2


def test_explain_linear_feature_re(newsgroups_train):
    clf = LogisticRegression(random_state=42)
    docs, y, target_names = newsgroups_train
    vec = CountVectorizer()
    X = vec.fit_transform(docs)
    clf.fit(X, y)
    res = explain_weights(clf, vec=vec, feature_re='^ath')
    for expl in format_as_all(res, clf):
        assert 'atheists' in expl
        assert 'atheism' in expl
        assert 'space' not in expl


@pytest.mark.parametrize(['clf'], [
    [RandomForestClassifier(n_estimators=100, random_state=42)],
    [ExtraTreesClassifier(n_estimators=100, random_state=24)],
    [GradientBoostingClassifier(random_state=42)],
    [AdaBoostClassifier(learning_rate=0.1, n_estimators=200, random_state=42)],
    [DecisionTreeClassifier(max_depth=3, random_state=42)],
])
def test_explain_random_forest(newsgroups_train, clf):
    docs, y, target_names = newsgroups_train
    vec = CountVectorizer()
    X = vec.fit_transform(docs)
    clf.fit(X.toarray(), y)

    get_res = lambda: explain_weights(clf, vec=vec, target_names=target_names,
                                      top=30)
    res = get_res()
    expl_text, expl_html = format_as_all(res, clf)
    for expl in [expl_text, expl_html]:
        assert 'feature importances' in expl
        assert 'god' in expl  # high-ranked feature

    if isinstance(clf, DecisionTreeClassifier):
        if _graphviz.is_supported():
            assert '<svg' in expl_html
        else:
            assert '<svg' not in expl_html

    assert res == get_res()


@pytest.mark.parametrize(['clf'], [
    [RandomForestClassifier(n_estimators=100, random_state=42)],
    [DecisionTreeClassifier(max_depth=3, random_state=42)],
])
def test_explain_random_forest_and_tree_feature_re(newsgroups_train, clf):
    docs, y, target_names = newsgroups_train
    vec = CountVectorizer()
    X = vec.fit_transform(docs)
    clf.fit(X.toarray(), y)
    res = explain_weights(
        clf, vec=vec, target_names=target_names, feature_re='^un')
    res.decision_tree = None  # does not respect feature_re
    for expl in format_as_all(res, clf):
        assert 'under' in expl
        assert 'god' not in expl  # filtered out


def test_explain_empty(newsgroups_train):
    clf = LogisticRegression(C=0.01, penalty='l1', random_state=42)
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    res = explain_weights(clf, vec=vec, target_names=target_names, top=20)
    format_as_all(res, clf)

    assert [t.target for t in res.targets] == target_names


def test_unsupported():
    vec = CountVectorizer()
    clf = BaseEstimator()
    res = explain_weights(clf, vec=vec)
    assert 'BaseEstimator' in res.error
    for expl in format_as_all(res, clf):
        assert 'Error' in expl
        assert 'BaseEstimator' in expl


@pytest.mark.parametrize(['clf'], [
    [ElasticNet(random_state=42)],
    [ElasticNetCV(random_state=42)],
    [Lars()],
    [Lasso(random_state=42)],
    [Ridge(random_state=42)],
    [RidgeCV()],
    [SGDRegressor(random_state=42)],
    [LinearRegression()],
    [LinearSVR(random_state=42)],
])
def test_explain_linear_regression(boston_train, clf):
    X, y, feature_names = boston_train
    clf.fit(X, y)
    res = explain_weights(clf)
    expl_text, expl_html = format_as_all(res, clf)

    for expl in [expl_text, expl_html]:
        assert 'x12' in expl
        assert 'x9' in expl
    assert '<BIAS>' in expl_text
    assert '&lt;BIAS&gt;' in expl_html

    pos, neg = top_pos_neg(res, 'y')
    assert 'x12' in pos or 'x12' in neg
    assert 'x9' in neg or 'x9' in pos
    assert '<BIAS>' in neg or '<BIAS>' in pos

    assert res == explain_weights(clf)


def test_explain_linear_regression_feature_re(boston_train):
    clf = ElasticNet(random_state=42)
    X, y, feature_names = boston_train
    clf.fit(X, y)
    res = explain_weights(clf, feature_re='^x[1-5]$')
    for expl in format_as_all(res, clf):
        assert 'x5' in expl
        assert 'x12' not in expl


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
    expl, _ = format_as_all(res, clf)

    assert 'x9' in expl
    assert '<BIAS>' in expl

    pos, neg = top_pos_neg(res, 'y2')
    assert 'x9' in neg or 'x9' in pos
    assert '<BIAS>' in neg or '<BIAS>' in pos

    assert res == explain_weights(clf)
