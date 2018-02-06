# -*- coding: utf-8 -*-
from __future__ import absolute_import
from functools import partial
import re

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_regression, make_multilabel_classification
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    HashingVectorizer
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
from sklearn.svm import (
    LinearSVC,
    LinearSVR,
    SVC,
    SVR,
    NuSVC,
    NuSVR,
    OneClassSVM,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.multiclass import OneVsRestClassifier
import pytest

from eli5 import _graphviz
from eli5 import explain_weights, explain_weights_sklearn
from eli5.sklearn.utils import has_intercept
from eli5.sklearn import InvertableHashingVectorizer
from .utils import format_as_all, get_all_features, get_names_coefs, SGD_KWARGS


def check_newsgroups_explanation_linear(
        clf, vec, target_names, explain_weights=explain_weights, binary=False,
        **kwargs):
    def get_result():
        _kwargs = dict(vec=vec, target_names=target_names, top=20)
        _kwargs.update(kwargs)
        return explain_weights(clf, **_kwargs)

    res = get_result()
    expl_text, expl_html = format_as_all(res, clf)

    got_targets = [cl.target for cl in res.targets]
    if binary:
        assert got_targets == [target_names[1]]
    else:
        assert got_targets == target_names

    _top = partial(top_pos_neg, res)

    pos, neg = _top('comp.graphics')
    assert 'file' in pos or 'graphics' in pos

    if not binary:
        for expl in [expl_text, expl_html]:
            assert 'comp.graphics' in expl
            assert 'atheists' in expl
            for label in target_names:
                assert str(label) in expl

        pos, neg = _top('alt.atheism')
        assert 'atheists' in pos

        pos, neg = _top('sci.space')
        assert 'space' in pos

        pos, neg = _top('talk.religion.misc')
        assert 'jesus' in pos or 'christians' in pos

    assert res == get_result()


def assert_explained_weights_linear_classifier(
        newsgroups_train, clf, add_bias=False, explain_weights=explain_weights,
        binary=False):
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()
    X = vec.fit_transform(docs)
    if add_bias:
        X = sp.hstack([X, np.ones((X.shape[0], 1))])
        feature_names = vec.get_feature_names() + ['BIAS']
    else:
        feature_names = None

    clf.fit(X, y)
    check_newsgroups_explanation_linear(clf, vec, target_names,
                                        feature_names=feature_names,
                                        explain_weights=explain_weights,
                                        binary=binary,
                                        top=(20, 20))


def assert_explained_weights_linear_regressor(boston_train, reg, has_bias=True):
    X, y, feature_names = boston_train
    reg.fit(X, y)
    res = explain_weights(reg)
    expl_text, expl_html = format_as_all(res, reg)

    for expl in [expl_text, expl_html]:
        assert 'x12' in expl
        assert 'x5' in expl

    if has_bias:
        assert '<BIAS>' in expl_text
        assert '&lt;BIAS&gt;' in expl_html

    pos, neg = top_pos_neg(res, 'y')
    assert 'x12' in pos or 'x12' in neg
    assert 'x5' in neg or 'x5' in pos

    if has_bias:
        assert '<BIAS>' in neg or '<BIAS>' in pos

    assert res == explain_weights(reg)


@pytest.mark.parametrize(['clf'], [
    [LogisticRegression(random_state=42)],
    [LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs')],
    [LogisticRegression(random_state=42, fit_intercept=False)],
    [LogisticRegressionCV(random_state=42)],
    [RidgeClassifier(random_state=42)],
    [RidgeClassifierCV()],
    [SGDClassifier(**SGD_KWARGS)],
    [SGDClassifier(loss='log', **SGD_KWARGS)],
    [PassiveAggressiveClassifier(random_state=42)],
    [Perceptron(random_state=42)],
    [LinearSVC(random_state=42)],
    [OneVsRestClassifier(SGDClassifier(**SGD_KWARGS))],
])
def test_explain_linear(newsgroups_train, clf):
    assert_explained_weights_linear_classifier(newsgroups_train, clf)


@pytest.mark.parametrize(['clf'], [
    [LogisticRegression(random_state=42)],
    [SGDClassifier(**SGD_KWARGS)],
    [SVC(kernel='linear', random_state=42)],
    [NuSVC(kernel='linear', random_state=42)],
])
def test_explain_linear_binary(newsgroups_train_binary, clf):
    assert_explained_weights_linear_classifier(newsgroups_train_binary, clf,
                                               binary=True)


@pytest.mark.parametrize(['clf'], [
    [SVC()],
    [NuSVC()],
    [SVR()],
    [NuSVR()],
])
def test_explain_linear_unsupported_kernels(clf):
    res = explain_weights(clf)
    assert 'supported' in res.error


@pytest.mark.parametrize(['clf'], [
    [SVC(kernel='linear')],
    [NuSVC(kernel='linear')],
])
def test_explain_linear_unsupported_multiclass(clf, newsgroups_train):
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()
    clf.fit(vec.fit_transform(docs), y)
    expl = explain_weights(clf, vec=vec)
    assert 'supported' in expl.error


def test_explain_one_class_svm():
    X = np.array([[0,0], [0, 1], [5, 3], [93, 94], [90, 91]])
    clf = OneClassSVM(kernel='linear', random_state=42).fit(X)
    res = explain_weights(clf)
    assert len(res.targets) == 1
    target = res.targets[0]
    assert target.target == '1'
    assert target.feature_weights.neg[0].feature == '<BIAS>'
    assert {f.feature for f in target.feature_weights.pos} == {'x1', 'x0'}
    for expl in format_as_all(res, clf):
        assert 'x1' in expl
        assert 'x0' in expl
        assert 'BIAS' in expl


def test_explain_one_class_svm_unsupported():
    X = np.array([[0,0], [0, 1], [5, 3], [93, 94], [90, 91]])
    clf = OneClassSVM().fit(X)
    expl = explain_weights(clf)
    assert 'supported' in expl.error


@pytest.mark.parametrize(['clf'], [
    [OneVsRestClassifier(SGDClassifier(**SGD_KWARGS))],
    [OneVsRestClassifier(LogisticRegression(random_state=42))],
])
def test_explain_linear_multilabel(clf):
    X, Y = make_multilabel_classification(random_state=42)
    clf.fit(X, Y)
    res = explain_weights_sklearn(clf)
    expl_text, expl_html = format_as_all(res, clf)
    for expl in [expl_text, expl_html]:
        assert 'y=4' in expl
        assert 'x0' in expl
        assert 'BIAS' in expl


@pytest.mark.parametrize(['clf'], [
    [LogisticRegression(random_state=42)],
    [LogisticRegression(random_state=42, fit_intercept=False)],
    [SGDClassifier(**SGD_KWARGS)],
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


@pytest.mark.parametrize(['vec'], [
    [CountVectorizer()],
    [HashingVectorizer(norm=None)],
])
def test_explain_linear_feature_filter(newsgroups_train, vec):
    clf = LogisticRegression(random_state=42)
    docs, y, target_names = newsgroups_train
    X = vec.fit_transform(docs)
    clf.fit(X, y)
    if isinstance(vec, HashingVectorizer):
        vec = InvertableHashingVectorizer(vec)
        vec.fit(docs)

    res = explain_weights(clf, vec=vec, feature_re='^ath')
    text_expl, _ = expls = format_as_all(res, clf)
    for expl in expls:
        assert 'atheists' in expl
        assert 'atheism' in expl
        assert 'space' not in expl
        assert 'BIAS' not in expl

    res = explain_weights(
        clf, vec=vec,
        feature_filter=lambda name: name.startswith('ath') or name == '<BIAS>')
    text_expl, _ = expls = format_as_all(res, clf)
    for expl in expls:
        assert 'atheists' in expl
        assert 'atheism' in expl
        assert 'space' not in expl
        assert 'BIAS' in expl
    assert '<BIAS>' in text_expl


@pytest.mark.parametrize(['clf'], [
    [RandomForestClassifier(n_estimators=100, random_state=42)],
    [ExtraTreesClassifier(n_estimators=100, random_state=24)],
    [GradientBoostingClassifier(random_state=42)],
    [AdaBoostClassifier(learning_rate=0.1, n_estimators=200, random_state=42)],
    [DecisionTreeClassifier(max_depth=3, random_state=42)],

    # FIXME:
    # [OneVsRestClassifier(DecisionTreeClassifier(max_depth=3, random_state=42))],
])
def test_explain_tree_classifier(newsgroups_train, clf, **explain_kwargs):
    docs, y, target_names = newsgroups_train
    vec = CountVectorizer()
    X = vec.fit_transform(docs)
    clf.fit(X.toarray(), y)
    assert_tree_classifier_explained(clf, vec, target_names, **explain_kwargs)


def assert_tree_classifier_explained(clf, vec, target_names, **explain_kwargs):
    top = 30
    get_res = lambda: explain_weights(
        clf, vec=vec, target_names=target_names, top=top, **explain_kwargs)
    res = get_res()
    expl_text, expl_html = format_as_all(res, clf)
    for expl in [expl_text, expl_html]:
        assert 'feature importances' in expl
        assert 'god' in expl  # high-ranked feature
        if len(res.feature_importances.importances) > top:
            assert 'more features' in expl or 'more &hellip;' in expl

    if isinstance(clf, (DecisionTreeClassifier, OneVsRestClassifier)):
        if _graphviz.is_supported():
            assert '<svg' in expl_html
        else:
            assert '<svg' not in expl_html

    assert res == get_res()


@pytest.mark.parametrize(['reg'], [
    [DecisionTreeRegressor(random_state=42)],
    [ExtraTreesRegressor(random_state=42)],
    [GradientBoostingRegressor(learning_rate=0.075, random_state=42)],
    [RandomForestRegressor(random_state=42)],
    [AdaBoostRegressor(random_state=42)],
])
def test_explain_tree_regressor(reg, boston_train):
    X, y, feature_names = boston_train
    reg.fit(X, y)
    res = explain_weights(reg, feature_names=feature_names)
    expl_text, expl_html = format_as_all(res, reg)
    for expl in [expl_text, expl_html]:
        assert 'BIAS' not in expl
        assert 'LSTAT' in expl

    if isinstance(reg, DecisionTreeRegressor):
        assert '---> 50' in expl_text


@pytest.mark.parametrize(['clf'], [
    [RandomForestClassifier(n_estimators=100, random_state=42)],
    [DecisionTreeClassifier(max_depth=3, random_state=42)],
])
def test_explain_random_forest_and_tree_feature_filter(newsgroups_train, clf):
    docs, y, target_names = newsgroups_train
    vec = CountVectorizer()
    X = vec.fit_transform(docs)
    clf.fit(X.toarray(), y)
    top = 30
    res = explain_weights(
        clf, vec=vec, target_names=target_names, feature_re='^a', top=top)
    res.decision_tree = None  # does not respect feature_filter
    for expl in format_as_all(res, clf):
        assert 'am' in expl
        assert 'god' not in expl  # filtered out
        if len(res.feature_importances.importances) > top:
            assert 'more features' in expl or 'more &hellip;' in expl


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
    with pytest.raises(TypeError):
        explain_weights(clf, unknown_argument=True)


@pytest.mark.parametrize(['reg'], [
    [ElasticNet(random_state=42)],
    [ElasticNetCV(random_state=42)],
    [HuberRegressor()],
    [Lars()],
    [LarsCV(max_n_alphas=10)],
    [Lasso(random_state=42)],
    [LassoCV(random_state=42)],
    [LassoLars(alpha=0.01)],
    [LassoLarsCV(max_n_alphas=10)],
    [LassoLarsIC()],
    [OrthogonalMatchingPursuit(n_nonzero_coefs=10)],
    [OrthogonalMatchingPursuitCV()],
    [PassiveAggressiveRegressor(C=0.1, random_state=42)],
    [Ridge(random_state=42)],
    [RidgeCV()],
    [SGDRegressor(**SGD_KWARGS)],
    [LinearRegression()],
    [LinearSVR(random_state=42)],
    [TheilSenRegressor(random_state=42)],
    [SVR(kernel='linear')],
    [NuSVR(kernel='linear')],
])
def test_explain_linear_regression(boston_train, reg):
    assert_explained_weights_linear_regressor(boston_train, reg)


@pytest.mark.parametrize(['reg'], [
    [Lasso(random_state=42)],
    [Lasso(fit_intercept=False, random_state=42)],
    [LinearRegression()],
    [LinearRegression(fit_intercept=False)],
    [SVR(kernel='linear')],
])
def test_explain_linear_regression_one_feature(reg):
    xs, ys = make_regression(n_samples=10, n_features=1, bias=7.5,
                             random_state=42)
    reg.fit(xs, ys)
    res = explain_weights(reg)
    expl_text, expl_html = format_as_all(res, reg)

    for expl in [expl_text, expl_html]:
        assert 'x0' in expl

    if has_intercept(reg):
        assert '<BIAS>' in expl_text
        assert '&lt;BIAS&gt;' in expl_html


def test_explain_linear_regression_feature_filter(boston_train):
    clf = ElasticNet(random_state=42)
    X, y, feature_names = boston_train
    clf.fit(X, y)
    res = explain_weights(clf, feature_names=feature_names,
                          feature_re=re.compile('ratio$', re.I))
    for expl in format_as_all(res, clf):
        assert 'PTRATIO' in expl
        assert 'LSTAT' not in expl


@pytest.mark.parametrize(['reg'], [
    [ElasticNet(random_state=42)],
    [Lars()],
    [Lasso(random_state=42)],
    [Ridge(random_state=42)],
    [LinearRegression()],
])
def test_explain_linear_regression_multitarget(reg):
    X, y = make_regression(n_samples=100, n_targets=3, n_features=10,
                           random_state=42)
    reg.fit(X, y)
    res = explain_weights(reg)
    expl, _ = format_as_all(res, reg)

    assert 'x9' in expl
    assert '<BIAS>' in expl

    pos, neg = top_pos_neg(res, 'y2')
    assert 'x9' in neg or 'x9' in pos
    assert '<BIAS>' in neg or '<BIAS>' in pos

    assert res == explain_weights(reg)


def test_explain_decision_tree_regressor_multitarget():
    X, y = make_regression(n_samples=100, n_targets=3, n_features=10,
                           random_state=42)
    reg = DecisionTreeRegressor(random_state=42, max_depth=3)
    reg.fit(X, y)
    res = explain_weights(reg)
    expl_text, expl_html = format_as_all(res, reg)

    assert 'x9' in expl_text
    assert '---> [' in expl_text
    assert '---> [[' not in expl_text

    assert res == explain_weights(reg)


@pytest.mark.parametrize(['clf'], [
    [DecisionTreeClassifier()],
    [ExtraTreesClassifier()],
])
def test_feature_importances_no_remaining(clf):
    """ Check that number of remaining features is shown if it is zero.
    """
    n = 100
    clf.fit(np.array([[i % 2 + 0.1 * np.random.random(), 0] for i in range(n)]),
            np.array([i % 2 for i in range(n)]))
    res = explain_weights(clf)
    for expl in format_as_all(res, clf):
        assert 'more features' not in expl and 'more &hellip;' not in expl


@pytest.mark.parametrize(['transformer', 'X', 'feature_names',
                          'explain_kwargs'], [
    [None, [[1, 0], [0, 1]], ['hello', 'world'], {}],
    [None, [[1, 0], [0, 1]], None,
     {'vec': CountVectorizer().fit(['hello', 'world'])}],
    [CountVectorizer(), ['hello', 'world'], None, {'top': 1}],
    [CountVectorizer(), ['hello', 'world'], None, {'top': 2}],
    [make_pipeline(CountVectorizer(),
                   SelectKBest(lambda X, y: np.array([3, 2, 1]), k=2)),
     ['hello', 'world zzzignored'], None, {}],
])
@pytest.mark.parametrize(['predictor'], [
    [LogisticRegression()],
    [LinearSVR()],
])
def test_explain_pipeline(predictor, transformer, X, feature_names,
                          explain_kwargs):
    y = [1, 0]
    expected = explain_weights(clone(predictor).fit([[1, 0], [0, 1]], y),
                               feature_names=['hello', 'world'],
                               **explain_kwargs)
    pipe = make_pipeline(transformer, clone(predictor)).fit(X, y)
    actual = explain_weights(pipe, feature_names=feature_names,
                             **explain_kwargs)
    assert expected._repr_html_() == actual._repr_html_()
