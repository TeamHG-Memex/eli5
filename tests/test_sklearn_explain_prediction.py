# -*- coding: utf-8 -*-
from __future__ import absolute_import
from functools import partial
from pprint import pprint

import pytest
from sklearn.datasets import make_regression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    Perceptron,
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
    TheilSenRegressor,
)
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from eli5 import explain_prediction
from eli5.formatters import format_as_text, fields
from eli5.sklearn.utils import has_intercept
from .utils import (
    format_as_all, strip_blanks, get_all_features, check_targets_scores)


format_as_all = partial(format_as_all, show_feature_values=True)


def assert_multiclass_linear_classifier_explained(newsgroups_train, clf,
                                                  explain_prediction):
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    get_res = lambda feature_flt=None: explain_prediction(
        clf, docs[0], vec=vec, target_names=target_names, top=20,
        feature_flt=feature_flt)
    res = get_res()
    pprint(res)
    expl_text, expl_html = format_as_all(res, clf)

    file_weight = None
    for e in res.targets:
        if e.target == 'comp.graphics':
            pos = get_all_features(e.feature_weights.pos, with_weights=True)
            assert 'file' in pos
            file_weight = pos['file']

    for expl in [expl_text, expl_html]:
        for label in target_names:
            assert str(label) in expl
        assert 'file' in expl

    assert res == get_res()

    flt_res = get_res(feature_flt=lambda name, _: name == 'file')
    format_as_all(flt_res, clf)
    for e in flt_res.targets:
        if e.target == 'comp.graphics':
            pos = get_all_features(e.feature_weights.pos, with_weights=True)
            assert 'file' in pos
            assert pos['file'] == file_weight
            assert len(pos) == 1


def assert_linear_regression_explained(boston_train, reg, explain_prediction,
                                       atol=1e-8, reg_has_intercept=None):
    X, y, feature_names = boston_train
    reg.fit(X, y)
    res = explain_prediction(reg, X[0], feature_names=feature_names)
    expl_text, expl_html = expls = format_as_all(res, reg)

    assert len(res.targets) == 1
    target = res.targets[0]
    assert target.target == 'y'
    get_pos_neg_features = lambda fw: (
        get_all_features(fw.pos, with_weights=True),
        get_all_features(fw.neg, with_weights=True))
    pos, neg = get_pos_neg_features(target.feature_weights)
    assert 'LSTAT' in pos or 'LSTAT' in neg

    if reg_has_intercept is None:
        reg_has_intercept = has_intercept(reg)
    if reg_has_intercept:
        assert '<BIAS>' in pos or '<BIAS>' in neg
        assert '<BIAS>' in expl_text
        assert '&lt;BIAS&gt;' in expl_html
    else:
        assert '<BIAS>' not in pos and '<BIAS>' not in neg
        assert '<BIAS>' not in expl_text
        assert 'BIAS' not in expl_html

    for expl in [expl_text, expl_html]:
        assert 'LSTAT' in expl
        assert '(score' in expl
    assert "'y'" in expl_text
    assert '<b>y</b>' in strip_blanks(expl_html)

    for expl in expls:
        assert_feature_values_present(expl, feature_names, X[0])

    assert res == explain_prediction(reg, X[0], feature_names=feature_names)
    check_targets_scores(res, atol=atol)

    flt_res = explain_prediction(reg, X[0], feature_names=feature_names,
                                 feature_flt=lambda name, v: name != 'LSTAT')
    format_as_all(flt_res, reg)
    flt_target = flt_res.targets[0]
    flt_pos, flt_neg = get_pos_neg_features(flt_target.feature_weights)
    assert 'LSTAT' not in flt_pos and 'LSTAT' not in flt_neg
    flt_all = dict(flt_pos, **flt_neg)
    expected = dict(pos, **neg)
    expected.pop('LSTAT')
    assert flt_all == expected


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
    check_targets_scores(res)


def assert_feature_values_present(expl, feature_names, x):
    assert 'Value' in expl
    any_features = False
    for feature, value in zip(feature_names, x):
        if feature in expl:
            assert '{:+.3f}'.format(value) in expl
            any_features = True
    assert any_features


@pytest.mark.parametrize(['clf'], [
    [LogisticRegression(random_state=42)],
    [LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs')],
    [LogisticRegression(random_state=42, fit_intercept=False)],
    [LogisticRegressionCV(random_state=42)],
    [SGDClassifier(random_state=42)],
    [SGDClassifier(loss='log', random_state=42)],
    [PassiveAggressiveClassifier(random_state=42)],
    [Perceptron(random_state=42)],
    [RidgeClassifier(random_state=42)],
    [RidgeClassifierCV()],
    [LinearSVC(random_state=42)],
    [OneVsRestClassifier(LogisticRegression(random_state=42))],
])
def test_explain_linear(newsgroups_train, clf):
    assert_multiclass_linear_classifier_explained(newsgroups_train, clf,
                                                  explain_prediction)


@pytest.mark.parametrize(['reg'], [
    [ElasticNet(random_state=42)],
    [ElasticNetCV(random_state=42)],
    [HuberRegressor()],
    [Lars()],
    [LarsCV(max_n_alphas=10)],
    [Lasso(random_state=42)],
    [LassoCV(n_alphas=10)],
    [LassoLars(alpha=0.1)],
    [LassoLarsCV(max_n_alphas=10)],
    [LassoLarsIC()],
    [LinearRegression()],
    [LinearSVR(random_state=42)],
    [OrthogonalMatchingPursuit(n_nonzero_coefs=10)],
    [OrthogonalMatchingPursuitCV()],
    [PassiveAggressiveRegressor(C=0.1)],
    [Ridge(random_state=42)],
    [RidgeCV()],
    [SGDRegressor(random_state=42)],
    [TheilSenRegressor()],
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


@pytest.mark.parametrize(['clf'], [
    [DecisionTreeClassifier()],
    [ExtraTreesClassifier()],
    [GradientBoostingClassifier(learning_rate=0.075)],
    [RandomForestClassifier()],
])
def test_explain_tree_clf_multiclass(clf, iris_train):
    X, y, feature_names, target_names = iris_train
    clf.fit(X, y)
    res = explain_prediction(
        clf, X[0], target_names=target_names, feature_names=feature_names)
    for expl in format_as_all(res, clf):
        for target in target_names:
            assert target in expl
        assert 'BIAS' in expl
        assert any(f in expl for f in feature_names)
        assert_feature_values_present(expl, feature_names, X[0])
    check_targets_scores(res)


@pytest.mark.parametrize(['clf'], [
    [DecisionTreeClassifier()],
    [ExtraTreesClassifier()],
    [GradientBoostingClassifier(learning_rate=0.075)],
    [RandomForestClassifier()],
])
def test_explain_tree_clf_binary(clf, iris_train_binary):
    X, y, feature_names = iris_train_binary
    clf.fit(X, y)
    res = explain_prediction(clf, X[0], feature_names=feature_names)
    format_as_all(res, clf)
    all_expls = []
    for x in X[:5]:
        res = explain_prediction(clf, x, feature_names=feature_names)
        text_expl = format_as_text(res, show=fields.WEIGHTS)
        print(text_expl)
        assert '<BIAS>' in text_expl
        check_targets_scores(res)
        all_expls.append(text_expl)
    assert any(f in ''.join(all_expls) for f in feature_names)


@pytest.mark.parametrize(['reg'], [
    [DecisionTreeRegressor()],
    [ExtraTreesRegressor()],
    [RandomForestRegressor()],
])
def test_explain_tree_regressor_multitarget(reg):
    X, y = make_regression(n_samples=100, n_targets=3, n_features=10,
                           random_state=42)
    reg.fit(X, y)
    res = explain_prediction(reg, X[0])
    for expl in format_as_all(res, reg):
        for target in ['y0', 'y1', 'y2']:
            assert target in expl
        assert 'BIAS' in expl
        assert any('x%d' % i in expl for i in range(10))
    check_targets_scores(res)


@pytest.mark.parametrize(['reg'], [
    [DecisionTreeRegressor()],
    [ExtraTreesRegressor()],
    [GradientBoostingRegressor(learning_rate=0.075)],
    [RandomForestRegressor()],
])
def test_explain_tree_regressor(reg, boston_train):
    X, y, feature_names = boston_train
    reg.fit(X, y)
    res = explain_prediction(reg, X[0], feature_names=feature_names)
    for expl in format_as_all(res, reg):
        assert_feature_values_present(expl, feature_names, X[0])
    all_expls = []
    for i, x in enumerate(X[:5]):
        res = explain_prediction(reg, x, feature_names=feature_names)
        text_expl = format_as_text(res, show=fields.WEIGHTS)
        print(text_expl)
        assert '<BIAS>' in text_expl
        check_targets_scores(res)
        all_expls.append(text_expl)
    assert any(f in ''.join(all_expls) for f in feature_names)


@pytest.mark.parametrize(['clf'], [
    [DecisionTreeClassifier()],
    [ExtraTreesClassifier()],
    [RandomForestClassifier()],
])
def test_explain_tree_classifier_text(clf, newsgroups_train_big):
    docs, y, target_names = newsgroups_train_big
    vec = CountVectorizer(binary=True, stop_words='english')
    X = vec.fit_transform(docs)
    clf.fit(X, y)
    res = explain_prediction(clf, docs[0], vec=vec, target_names=target_names)
    check_targets_scores(res)
    format_as_all(res, clf)
