# -*- coding: utf-8 -*-
from __future__ import absolute_import
from functools import partial
from pprint import pprint
import re
from typing import List

import pytest
import numpy as np
from sklearn.base import BaseEstimator
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
from sklearn.svm import (
    LinearSVC,
    LinearSVR,
    SVC,
    SVR,
    NuSVC,
    NuSVR,
    OneClassSVM,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from eli5 import explain_prediction, explain_prediction_sklearn
from eli5.base import Explanation
from eli5.formatters import format_as_text, fields
from eli5.sklearn.utils import has_intercept
from .utils import (
    format_as_all, strip_blanks, get_all_features, check_targets_scores,
    SGD_KWARGS)


format_as_all = partial(format_as_all, show_feature_values=True)


def assert_multiclass_linear_classifier_explained(newsgroups_train, clf,
                                                  explain_prediction):
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    get_res = lambda **kwargs: explain_prediction(
        clf, docs[0], vec=vec, target_names=target_names, top=20, **kwargs)
    res = get_res()
    pprint(res)
    expl_text, expl_html = format_as_all(res, clf)

    file_weight = None
    scores = {}
    for e in res.targets:
        scores[e.target] = e.score
        if e.target == 'comp.graphics':
            pos = get_all_features(e.feature_weights.pos, with_weights=True)
            assert 'file' in pos
            file_weight = pos['file']

    for expl in [expl_text, expl_html]:
        for label in target_names:
            assert str(label) in expl
        assert 'file' in expl

    assert res == get_res()

    flt_res = get_res(feature_re='^file$')
    format_as_all(flt_res, clf)
    for e in flt_res.targets:
        if e.target == 'comp.graphics':
            pos = get_all_features(e.feature_weights.pos, with_weights=True)
            assert 'file' in pos
            assert pos['file'] == file_weight
            assert len(pos) == 1

    targets_res = get_res(targets=['comp.graphics'])
    assert len(targets_res.targets) == 1
    assert targets_res.targets[0].target == 'comp.graphics'

    top2_targets_res = get_res(top_targets=2)
    assert len(top2_targets_res.targets) == 2
    sorted_targets = [
        t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    assert [t.target for t in top2_targets_res.targets] == sorted_targets[:2]

    top_neg_targets_res = get_res(top_targets=-1)
    assert len(top_neg_targets_res.targets) == 1
    assert [t.target for t in top_neg_targets_res.targets] == sorted_targets[-1:]


def assert_binary_linear_classifier_explained(newsgroups_train_binary, clf,
                                              explain_prediction):
    docs, y, target_names = newsgroups_train_binary
    vec = TfidfVectorizer()

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    assert y[2] == 1
    cg_document = docs[2]
    res = explain_prediction(clf, cg_document, vec=vec,
                             target_names=target_names, top=20)
    expl_text, expl_html = format_as_all(res, clf)
    for expl in [expl_text, expl_html]:
        assert 'software' in expl or 'thanks' in expl
        assert target_names[1] in expl

    assert y[15] == 0
    atheism_document = docs[15]
    res = explain_prediction(clf, atheism_document, vec=vec,
                             target_names=target_names, top=20)
    expl_text, expl_html = format_as_all(res, clf)
    for expl in [expl_text, expl_html]:
        assert 'god' in expl
        assert target_names[0] in expl

    assert_correct_class_explained_binary(clf, X[::10])


def assert_linear_regression_explained(boston_train, reg, explain_prediction,
                                       atol=1e-8, reg_has_intercept=None):
    X, y, feature_names = boston_train
    reg.fit(X, y)
    assert_trained_linear_regression_explained(
        X[0], feature_names, reg, explain_prediction,
        atol=atol, reg_has_intercept=reg_has_intercept)


def assert_trained_linear_regression_explained(
        x, feature_names, reg, explain_prediction,
        atol=1e-8, reg_has_intercept=None):
    res = explain_prediction(reg, x, feature_names=feature_names)
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
        assert_feature_values_present(expl, feature_names, x)

    assert res == explain_prediction(reg, x, feature_names=feature_names)
    check_targets_scores(res, atol=atol)

    flt_res = explain_prediction(reg, x, feature_names=feature_names,
                                 feature_filter=lambda name, v: name != 'LSTAT')
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
    if has_intercept(reg):
        assert '<BIAS>' in pos or '<BIAS>' in neg

    assert 'x8' in expl_text
    if has_intercept(reg):
        assert '<BIAS>' in expl_text
    assert "'y2'" in expl_text

    assert res == explain_prediction(reg, X[0])
    check_targets_scores(res)

    top_targets_res = explain_prediction(reg, X[0], top_targets=1)
    assert len(top_targets_res.targets) == 1


def assert_correct_class_explained_binary(clf, X):
    size = X.shape[0]
    expl_predicted = assert_predicted_class_used(clf, X)
    expl_target0 = assert_class_used(clf, X, np.zeros(size), targets=[0])
    expl_target1 = assert_class_used(clf, X, np.ones(size), targets=[1])

    get_fw = lambda expl: expl.targets[0].feature_weights
    for pred, t0, t1 in zip(expl_predicted, expl_target0, expl_target1):
        # sanity check
        assert get_fw(pred).pos == get_fw(pred).pos
        y_pred = pred.targets[0].target
        y_1 = t1.targets[0].target

        if y_pred == y_1:
            # targets=[True] result is the same as for the predicted class
            # if predicted class is True
            assert get_fw(pred).pos == get_fw(t1).pos
            assert get_fw(pred).neg == get_fw(t1).neg
        else:
            # targets=[False] result is the same as for the predicted class
            # if predicted class is False
            assert get_fw(pred).pos == get_fw(t0).pos
            assert get_fw(pred).neg == get_fw(t0).neg

        # explanations shouldn't be the same for positive and negative classes
        assert get_fw(t0).pos != get_fw(t1).pos
        assert get_fw(t0).neg != get_fw(t1).neg


def assert_predicted_class_used(clf, X):
    """ Check that predicted classes are used for explanations
    of X predictions """
    y_pred = clf.predict(X)
    return assert_class_used(clf, X, y_pred)


def assert_class_used(clf, X, y, **explain_kwargs):
    # type: (...) -> List[Explanation]
    """ Check that classes y are used for explanations of X predictions """
    explanations = []
    for x, pred_target in zip(X, y):
        res = explain_prediction(clf, x, **explain_kwargs)  # type: Explanation
        explanations.append(res)
        assert len(res.targets) == 1
        if res.targets[0].score != 0:
            assert res.targets[0].target == pred_target
    return explanations


def assert_feature_values_present(expl, feature_names, x):
    assert 'Value' in expl
    any_features = False
    for feature, value in zip(feature_names, x):
        if re.search(r'[\b\s]{}[\b\s]'.format(re.escape(feature)), expl):
            assert '{:.3f}'.format(value) in expl, feature
            any_features = True
    assert any_features


def assert_explain_prediction_single_target(estimator, X, feature_names):
    get_res = lambda _x, **kwargs: explain_prediction(
        estimator, _x, feature_names=feature_names, **kwargs)
    res = get_res(X[0])
    for expl in format_as_all(res, estimator):
        assert_feature_values_present(expl, feature_names, X[0])

    # take first elements in the dataset; check that
    # 1. <BIAS> feature is present;
    # 2. scores have correct absolute values;
    # 3. feature filter function works.
    checked_flt = False
    all_expls = []
    for x in X[:5]:
        res = get_res(x)
        text_expl = format_as_text(res, show=fields.WEIGHTS)
        print(text_expl)
        assert '<BIAS>' in text_expl
        check_targets_scores(res)
        all_expls.append(text_expl)
        checked_flt = checked_flt or _assert_feature_filter_works(get_res, x)

    assert checked_flt
    assert any(f in ''.join(all_expls) for f in feature_names)


def _assert_feature_filter_works(get_res, x):
    res = get_res(x)
    get_all = lambda fw: get_all_features(fw.pos) | get_all_features(fw.neg)
    all_features = get_all(res.targets[0].feature_weights)
    if len(all_features) > 1:
        f = list(all_features - {'<BIAS>'})[0]
        flt_res = get_res(x, feature_filter=lambda name, _: name != f)
        flt_features = get_all(flt_res.targets[0].feature_weights)
        assert flt_features == (all_features - {f})
        return True
    return False


@pytest.mark.parametrize(['clf'], [
    [LogisticRegression(random_state=42)],
    [LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs')],
    [LogisticRegression(random_state=42, fit_intercept=False)],
    [LogisticRegressionCV(random_state=42)],
    [SGDClassifier(**SGD_KWARGS)],
    [SGDClassifier(loss='log', **SGD_KWARGS)],
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
    if isinstance(clf, OneVsRestClassifier):
        assert_multiclass_linear_classifier_explained(
            newsgroups_train, clf, explain_prediction_sklearn)


@pytest.mark.parametrize(['clf'], [
    [LogisticRegression(random_state=42)],
    [LogisticRegressionCV(random_state=42)],
    [OneVsRestClassifier(LogisticRegression(random_state=42))],
    [SGDClassifier(**SGD_KWARGS)],
    [SVC(kernel='linear', random_state=42)],
    [SVC(kernel='linear', random_state=42, decision_function_shape='ovr')],
    [SVC(kernel='linear', random_state=42, decision_function_shape='ovr',
         probability=True)],
    [SVC(kernel='linear', random_state=42, probability=True)],
    [NuSVC(kernel='linear', random_state=42)],
    [NuSVC(kernel='linear', random_state=42, decision_function_shape='ovr')],
])
def test_explain_linear_binary(newsgroups_train_binary, clf):
    assert_binary_linear_classifier_explained(newsgroups_train_binary, clf,
                                              explain_prediction)


def test_explain_one_class_svm():
    X = np.array([[0, 0], [0, 1], [5, 3], [93, 94], [90, 91]])
    clf = OneClassSVM(kernel='linear', random_state=42).fit(X)
    res = explain_prediction(clf, X[0])
    assert res.targets[0].score < 0
    for expl in format_as_all(res, clf):
        assert 'BIAS' in expl
        assert 'x0' not in expl
        assert 'x1' not in expl

    res = explain_prediction(clf, X[4])
    assert res.targets[0].score > 0
    for expl in format_as_all(res, clf):
        assert 'BIAS' in expl
        assert 'x0' in expl
        assert 'x1' in expl


@pytest.mark.parametrize(['clf'], [
    [SVC()],
    [NuSVC()],
    [OneClassSVM()],
])
def test_explain_linear_classifiers_unsupported_kernels(clf, newsgroups_train_binary):
    docs, y, target_names = newsgroups_train_binary
    vec = TfidfVectorizer()
    clf.fit(vec.fit_transform(docs), y)
    res = explain_prediction(clf, docs[0], vec=vec)
    assert 'supported' in res.error


@pytest.mark.parametrize(['clf'], [
    [SVC(kernel='linear')],
    [NuSVC(kernel='linear')],
])
def test_explain_linear_unsupported_multiclass(clf, newsgroups_train):
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()
    clf.fit(vec.fit_transform(docs), y)
    expl = explain_prediction(clf, docs[0], vec=vec)
    assert 'supported' in expl.error


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
    [LinearRegression(fit_intercept=False)],
    [LinearSVR(random_state=42)],
    [OrthogonalMatchingPursuit(n_nonzero_coefs=10)],
    [OrthogonalMatchingPursuitCV()],
    [PassiveAggressiveRegressor(C=0.1)],
    [Ridge(random_state=42)],
    [RidgeCV()],
    [SGDRegressor(**SGD_KWARGS)],
    [TheilSenRegressor()],
    [SVR(kernel='linear')],
    [NuSVR(kernel='linear')],
])
def test_explain_linear_regression(boston_train, reg):
    assert_linear_regression_explained(boston_train, reg, explain_prediction)


@pytest.mark.parametrize(['reg'], [
    [SVR()],
    [NuSVR()],
])
def test_explain_libsvm_linear_regressors_unsupported_kernels(reg, boston_train):
    X, y, feature_names = boston_train
    reg.fit(X, y)
    res = explain_prediction(reg, X[0], feature_names=feature_names)
    assert 'supported' in res.error


@pytest.mark.parametrize(['reg'], [
    [ElasticNet(random_state=42)],
    [Lars()],
    [Lasso(random_state=42)],
    [LinearRegression()],
    [LinearRegression(fit_intercept=False)],
    [Ridge(random_state=42)],
])
def test_explain_linear_regression_multitarget(reg):
    assert_multitarget_linear_regression_explained(reg, explain_prediction)


@pytest.mark.parametrize(['clf'], [
    [DecisionTreeClassifier(random_state=42)],
    [ExtraTreesClassifier(random_state=42)],
    [GradientBoostingClassifier(learning_rate=0.075, random_state=42)],
    [RandomForestClassifier(random_state=42)],
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

    top_targets_res = explain_prediction(clf, X[0], top_targets=1)
    assert len(top_targets_res.targets) == 1


@pytest.mark.parametrize(['clf'], [
    [DecisionTreeClassifier(random_state=42)],
    [ExtraTreesClassifier(random_state=42)],
    [GradientBoostingClassifier(learning_rate=0.075, random_state=42)],
    [RandomForestClassifier(random_state=42)],
    [LogisticRegression(random_state=42)],
    [OneVsRestClassifier(LogisticRegression(random_state=42))],
    [SGDClassifier(**SGD_KWARGS)],
    [SVC(kernel='linear', random_state=42)],
    [NuSVC(kernel='linear', random_state=42)],
])
def test_explain_clf_binary_iris(clf, iris_train_binary):
    X, y, feature_names = iris_train_binary
    clf.fit(X, y)
    assert_explain_prediction_single_target(clf, X, feature_names)
    assert_correct_class_explained_binary(clf, X)


@pytest.mark.parametrize(['reg'], [
    [DecisionTreeRegressor(random_state=42)],
    [ExtraTreesRegressor(random_state=42)],
    [RandomForestRegressor(random_state=42)],
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

    top_targets_res = explain_prediction(reg, X[0], top_targets=1)
    assert len(top_targets_res.targets) == 1


@pytest.mark.parametrize(['reg'], [
    [DecisionTreeRegressor(random_state=42)],
    [ExtraTreesRegressor(random_state=42)],
    [GradientBoostingRegressor(learning_rate=0.075, random_state=42)],
    [RandomForestRegressor(random_state=42)],
])
def test_explain_tree_regressor(reg, boston_train):
    X, y, feature_names = boston_train
    reg.fit(X, y)
    assert_explain_prediction_single_target(reg, X, feature_names)


@pytest.mark.parametrize(['clf'], [
    [DecisionTreeClassifier(random_state=42)],
    [ExtraTreesClassifier(random_state=42)],
    [RandomForestClassifier(random_state=42)],
])
def test_explain_tree_classifier_text(clf, newsgroups_train_big):
    docs, y, target_names = newsgroups_train_big
    vec = CountVectorizer(binary=True, stop_words='english')
    X = vec.fit_transform(docs)
    clf.fit(X, y)
    res = explain_prediction(clf, docs[0], vec=vec, target_names=target_names)
    check_targets_scores(res)
    format_as_all(res, clf)


def test_unsupported():
    vec = CountVectorizer()
    clf = BaseEstimator()
    doc = 'doc'
    res = explain_prediction(clf, doc, vec=vec)
    assert 'BaseEstimator' in res.error
    for expl in format_as_all(res, clf):
        assert 'Error' in expl
        assert 'BaseEstimator' in expl
    with pytest.raises(TypeError):
        explain_prediction(clf, doc, unknown_argument=True)


@pytest.mark.parametrize(['reg'], [
    [LinearRegression()],
    [LinearRegression(fit_intercept=False)],
    [RandomForestRegressor(random_state=42)],
])
def test_explain_prediction_pandas(reg, boston_train):
    pd = pytest.importorskip('pandas')
    X, y, feature_names = boston_train
    df = pd.DataFrame(X, columns=feature_names)
    reg.fit(df, y)
    res = explain_prediction(reg, df.iloc[0])
    for expl in format_as_all(res, reg):
        assert 'PTRATIO' in expl
        if has_intercept(reg):
            assert 'BIAS' in expl
