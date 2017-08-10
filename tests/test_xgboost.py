# -*- coding: utf-8 -*-
from __future__ import absolute_import

import pytest
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
pytest.importorskip('xgboost')
import xgboost
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from eli5.xgboost import (
    _parse_tree_dump, _xgb_n_targets, _missing_values_set_to_nan,
    _parent_value, _parse_dump_line, _check_booster_args,
)
from eli5.explain import explain_prediction, explain_weights
from eli5.formatters.text import format_as_text
from eli5.formatters import fields
from .utils import format_as_all, get_all_features, check_targets_scores
from .test_sklearn_explain_weights import (
    test_explain_tree_classifier as _check_rf_classifier,
    test_explain_random_forest_and_tree_feature_filter as _check_rf_feature_filter,
    test_feature_importances_no_remaining as _check_rf_no_remaining,
    assert_tree_classifier_explained,
)
from .test_sklearn_explain_prediction import (
    assert_linear_regression_explained,
    assert_trained_linear_regression_explained,
    assert_explain_prediction_single_target,
    test_explain_clf_binary_iris as _check_binary_classifier,
    test_explain_prediction_pandas as _check_explain_prediction_pandas,
)


@pytest.mark.parametrize(['importance_type'], [['gain'], ['weight'], ['cover']])
def test_explain_xgboost(newsgroups_train, importance_type):
    _check_rf_classifier(newsgroups_train, XGBClassifier(n_estimators=10),
                         importance_type=importance_type)


def test_explain_booster(newsgroups_train):
    docs, y, target_names = newsgroups_train
    vec = CountVectorizer()
    X = vec.fit_transform(docs)
    booster = xgboost.train(
        params={'objective': 'multi:softprob', 'silent': True, 'max_depth': 3,
                'num_class': len(target_names)},
        dtrain=xgboost.DMatrix(X, label=y, missing=np.nan),
        num_boost_round=10)
    assert_tree_classifier_explained(booster, vec, target_names)


def test_explain_xgboost_feature_filter(newsgroups_train):
    _check_rf_feature_filter(newsgroups_train, XGBClassifier(n_estimators=10))


def test_feature_importances_no_remaining():
    _check_rf_no_remaining(XGBClassifier(n_estimators=10))


def test_explain_xgboost_regressor(boston_train):
    xs, ys, feature_names = boston_train
    reg = XGBRegressor()
    reg.fit(xs, ys)
    res = explain_weights(reg)
    for expl in format_as_all(res, reg):
        assert 'f12' in expl
    res = explain_weights(reg, feature_names=feature_names)
    for expl in format_as_all(res, reg):
        assert 'LSTAT' in expl


def test_explain_xgboost_booster(boston_train):
    xs, ys, feature_names = boston_train
    booster = xgboost.train(
        params={'objective': 'reg:linear', 'silent': True},
        dtrain=xgboost.DMatrix(xs, label=ys),
    )
    res = explain_weights(booster)
    for expl in format_as_all(res, booster):
        assert 'f12' in expl
    res = explain_weights(booster, feature_names=feature_names)
    for expl in format_as_all(res, booster):
        assert 'LSTAT' in expl


@pytest.mark.parametrize(
    ['missing', 'use_booster'],
    [[np.nan, False], [0, False], [np.nan, True]])
def test_explain_prediction_clf_binary(
        newsgroups_train_binary_big, missing, use_booster):
    docs, ys, target_names = newsgroups_train_binary_big
    vec = CountVectorizer(stop_words='english')
    xs = vec.fit_transform(docs)
    explain_kwargs = {}
    if use_booster:
        clf = xgboost.train(
            params={'objective': 'binary:logistic',
                    'silent': True,
                    'max_depth': 2},
            dtrain=xgboost.DMatrix(xs, label=ys, missing=missing),
            num_boost_round=100,
        )
        explain_kwargs.update({'missing': missing, 'is_regression': False})
    else:
        clf = XGBClassifier(n_estimators=100, max_depth=2, missing=missing)
        clf.fit(xs, ys)
    get_res = lambda **kwargs: explain_prediction(
        clf, 'computer graphics in space: a sign of atheism',
        vec=vec, target_names=target_names, **dict(kwargs, **explain_kwargs))
    res = get_res()
    for expl in format_as_all(res, clf, show_feature_values=True):
        assert 'graphics' in expl
        assert 'Missing' in expl
    check_targets_scores(res)
    weights = res.targets[0].feature_weights
    pos_features = get_all_features(weights.pos)
    neg_features = get_all_features(weights.neg)
    assert 'graphics' in pos_features
    assert 'computer' in pos_features
    assert 'atheism' in neg_features

    flt_res = get_res(feature_re='gra')
    flt_pos_features = get_all_features(flt_res.targets[0].feature_weights.pos)
    assert 'graphics' in flt_pos_features
    assert 'computer' not in flt_pos_features

    flt_value_res = get_res(feature_filter=lambda _, v: not np.isnan(v))
    for expl in format_as_all(flt_value_res, clf, show_feature_values=True):
        assert 'Missing' not in expl


@pytest.mark.parametrize(['clf'], [
    [XGBClassifier(n_estimators=50)],
    [XGBRegressor(n_estimators=50)],
])
def test_explain_prediction_xgboost_binary_iris(clf, iris_train_binary):
    X, y, feature_names = iris_train_binary
    clf.fit(X, y)
    assert_explain_prediction_single_target(clf, X, feature_names)


def test_explain_prediction_xgboost_clf_binary_iris(iris_train_binary):
    clf = XGBClassifier(n_estimators=50)
    _check_binary_classifier(clf, iris_train_binary)


@pytest.mark.parametrize(
    ['filter_missing', 'use_booster'],
    [[True, False], [False, False], [True, True]])
def test_explain_prediction_clf_multitarget(
        newsgroups_train, filter_missing, use_booster):
    docs, ys, target_names = newsgroups_train
    vec = CountVectorizer(stop_words='english')
    xs = vec.fit_transform(docs)
    if use_booster:
        clf = xgboost.train(
            params={'objective': 'multi:softprob',
                    'num_class': len(target_names),
                    'silent': True,
                    'max_depth': 2},
            dtrain=xgboost.DMatrix(xs, label=ys, missing=np.nan),
            num_boost_round=100,
        )
    else:
        clf = XGBClassifier(n_estimators=100, max_depth=2)
        clf.fit(xs, ys)
    feature_filter = (lambda _, v: not np.isnan(v)) if filter_missing else None
    doc = 'computer graphics in space: a new religion'
    res = explain_prediction(clf, doc, vec=vec, target_names=target_names,
                             feature_filter=feature_filter)
    format_as_all(res, clf)
    if not filter_missing:
        check_targets_scores(res)
    graphics_weights = res.targets[1].feature_weights
    assert 'computer' in get_all_features(graphics_weights.pos)
    religion_weights = res.targets[3].feature_weights
    assert 'religion' in get_all_features(religion_weights.pos)

    top_target_res = explain_prediction(clf, doc, vec=vec, top_targets=2)
    assert len(top_target_res.targets) == 2
    assert sorted(t.proba for t in top_target_res.targets) == sorted(
        t.proba for t in res.targets)[-2:]


def test_explain_prediction_clf_xor():
    true_xs = [[np.random.randint(2), np.random.randint(2)] for _ in range(100)]
    xs = np.array([[np.random.normal(x, 0.2), np.random.normal(y, 0.2)]
                   for x, y in true_xs])
    ys = np.array([x == y for x, y in true_xs])
    clf = XGBClassifier(n_estimators=100, max_depth=2)
    clf.fit(xs, ys)
    res = explain_prediction(clf, np.array([1, 1]))
    format_as_all(res, clf)
    for x in [[0, 1], [1, 0], [0, 0], [1, 1]]:
        res = explain_prediction(clf, np.array(x))
        print(x)
        print(format_as_text(res, show=fields.WEIGHTS))
        check_targets_scores(res)


def test_dense_missing():
    xs = np.array([[0, 1], [0, 2], [1, 2], [1, 0], [0.1, 0.1]] * 10)
    ys = np.array([0, 0, 3, 2, 0.2] * 10)
    # set too high n_estimators to check empty trees too
    reg = XGBRegressor(n_estimators=100, max_depth=2, missing=0)
    reg.fit(xs, ys)
    res = explain_prediction(reg, np.array([2, 0]))
    check_targets_scores(res)
    for expl in format_as_all(res, reg, show_feature_values=True):
        assert 'x0' in expl
        assert 'x1' in expl
        assert 'Missing' in expl
    flt_res = explain_prediction(reg, np.array([2, 0]),
                                 feature_filter=lambda _, v: not np.isnan(v))
    for expl in format_as_all(flt_res, reg, show_feature_values=True):
        assert 'x1' not in expl
        assert 'Missing' not in expl


def test_explain_prediction_clf_interval():
    true_xs = [[np.random.randint(3), np.random.randint(10)]
               for _ in range(1000)]
    xs = np.array([[np.random.normal(x, 0.2), np.random.normal(y, 0.2)]
                   for x, y in true_xs])
    ys = np.array([x == 1 for x, _ in true_xs])
    clf = XGBClassifier(n_estimators=100, max_depth=2)
    clf.fit(xs, ys)
    res = explain_prediction(clf, np.array([1.23, 1.45]))
    for expl in format_as_all(res, clf, show_feature_values=True):
        assert 'x0' in expl
        assert '1.23' in expl
    for x in [[0, 1], [1, 1], [2, 1], [0.8, 5], [1.2, 5]]:
        res = explain_prediction(clf, np.array(x))
        print(x)
        print(format_as_text(res, show=fields.WEIGHTS))
        check_targets_scores(res)


def test_explain_prediction_reg(boston_train):
    assert_linear_regression_explained(
        boston_train, XGBRegressor(), explain_prediction,
        reg_has_intercept=True)


def test_explain_prediction_reg_booster(boston_train):
    X, y, feature_names = boston_train
    booster = xgboost.train(
        params={'objective': 'reg:linear', 'silent': True, 'max_depth': 2},
        dtrain=xgboost.DMatrix(X, label=y),
    )
    assert_trained_linear_regression_explained(
        X[0], feature_names, booster, explain_prediction,
        reg_has_intercept=True)


def test_explain_prediction_feature_union_dense():
    # Test FeatureUnion handling and missing features in dense matrix
    transformer = lambda key: FunctionTransformer(
        lambda xs: np.array([[x.get(key, np.nan)] for x in xs]),
        validate=False)
    vec = FeatureUnion([('x', transformer('x')), ('y', transformer('y'))])
    gauss = np.random.normal
    data = [(gauss(1), 2 + 10 * gauss(1)) for _ in range(200)]
    ys = [-3 * x + y for x, y in data]
    xs = [{'x': gauss(x), 'y': gauss(y)} for x, y in data]
    for x in xs[:50]:
        del x['x']
    for x in xs[-50:]:
        del x['y']
    reg = XGBRegressor()
    reg.fit(vec.transform(xs), ys)
    res = explain_prediction(reg, xs[0], vec=vec, feature_names=['_x_', '_y_'])
    check_targets_scores(res)
    for expl in format_as_all(res, reg, show_feature_values=True):
        assert 'Missing' in expl
        assert '_y_' in expl
        assert '_x_' in expl


def test_explain_prediction_feature_union_sparse(newsgroups_train_binary):
    # FeatureUnion with sparce features and text highlighting
    docs, ys, target_names = newsgroups_train_binary
    vec = FeatureUnion([
        ('word', CountVectorizer(stop_words='english')),
        ('char', CountVectorizer(ngram_range=(3, 3))),
        ])
    clf = XGBClassifier(n_estimators=100, max_depth=2, missing=0)
    xs = vec.fit_transform(docs)
    clf.fit(xs, ys)
    res = explain_prediction(
        clf, 'computer graphics in space: a sign of atheism',
        vec=vec, target_names=target_names)
    format_as_all(res, clf)
    check_targets_scores(res)
    weights = res.targets[0].feature_weights
    pos_features = get_all_features(weights.pos)
    assert 'word__graphics' in pos_features
    assert res.targets[0].weighted_spans


def test_explain_prediction_pandas(boston_train):
    _check_explain_prediction_pandas(XGBRegressor(), boston_train)


def test_explain_weights_feature_names_pandas(boston_train):
    pd = pytest.importorskip('pandas')
    X, y, feature_names = boston_train
    df = pd.DataFrame(X, columns=feature_names)
    reg = XGBRegressor().fit(df, y)

    # it shoud pick up feature names from DataFrame columns
    res = explain_weights(reg)
    for expl in format_as_all(res, reg):
        assert 'PTRATIO' in expl

    # it is possible to override DataFrame feature names
    numeric_feature_names = ["zz%s" % idx for idx in range(len(feature_names))]
    res = explain_weights(reg, feature_names=numeric_feature_names)
    for expl in format_as_all(res, reg):
        assert 'zz12' in expl


def test_explain_prediction_pandas_dot_in_feature_name(boston_train):
    pd = pytest.importorskip('pandas')
    X, y, feature_names = boston_train
    feature_names = ["%s.%s" % (name, idx)
                     for idx, name in enumerate(feature_names)]
    df = pd.DataFrame(X, columns=feature_names)

    reg = XGBRegressor()
    reg.fit(df, y)
    res = explain_prediction(reg, df.iloc[0])
    for expl in format_as_all(res, reg):
        assert 'PTRATIO.1' in expl


def test_parse_tree_dump():
    text_dump = '''\
0:[f1793<-9.53674e-07] yes=1,no=2,missing=1,gain=6.112,cover=37.5
	1:[f371<-9.53674e-07] yes=3,no=4,missing=3,gain=4.09694,cover=28.5
		3:leaf=-0.0396476,cover=27.375
		4:leaf=0.105882,cover=1.125
	2:[f3332<-9.53674e-07] yes=5,no=6,missing=5,gain=3.41271,cover=9
		5:leaf=0.0892308,cover=7.125
		6:leaf=-0.0434783,cover=1.875
'''
    assert _parse_tree_dump(text_dump) == {
        'children': [
            {'children': [{'cover': 27.375, 'leaf': -0.0396476, 'nodeid': 3},
                          {'cover': 1.125, 'leaf': 0.105882, 'nodeid': 4}],
             'cover': 28.5,
             'depth': 1,
             'gain': 4.09694,
             'missing': 3,
             'no': 4,
             'nodeid': 1,
             'split': 'f371',
             'split_condition': -9.53674e-07,
             'yes': 3},
            {'children': [{'cover': 7.125, 'leaf': 0.0892308, 'nodeid': 5},
                          {'cover': 1.875, 'leaf': -0.0434783, 'nodeid': 6}],
             'cover': 9.0,
             'depth': 1,
             'gain': 3.41271,
             'missing': 5,
             'no': 6,
             'nodeid': 2,
             'split': 'f3332',
             'split_condition': -9.53674e-07,
             'yes': 5}],
        'cover': 37.5,
        'depth': 0,
        'gain': 6.112,
        'missing': 1,
        'no': 2,
        'nodeid': 0,
        'split': 'f1793',
        'split_condition': -9.53674e-07,
        'yes': 1}

    with pytest.raises(ValueError):
        _parse_tree_dump('foo')

    with pytest.raises(ValueError):
        _parse_tree_dump('''\
0:[f1793<-9.53674e-07] yes=1,no=2,missing=1,gain=6.112,cover=37.5
		1:[f371<-9.53674e-07] yes=3,no=4,missing=3,gain=4.09694,cover=28.5
''')


@pytest.mark.parametrize(
    ['line', 'result'],
    [
        (
            '0:[LSTAT.12<7.3] yes=1,no=2,missing=1,gain=4246.13,cover=100',
            (0, {
                'depth': 0,
                'nodeid': 0,
                'split': 'LSTAT.12',
                'split_condition': 7.3,
                'yes': 1,
                'no': 2,
                'missing': 1,
                'gain': 4246.13,
                'cover': 100,
            })
        ),
    ])
def test_parse_dump_line(line, result):
    assert _parse_dump_line(line) == result


def test_xgb_n_targets():
    clf = XGBClassifier()
    clf.fit(np.array([[0], [1]]), np.array([0, 1]))
    assert _xgb_n_targets(clf) == 1

    clf = XGBClassifier()
    clf.fit(np.array([[0], [1], [2]]), np.array([0, 1, 2]))
    assert _xgb_n_targets(clf) == 3

    reg = XGBRegressor()
    reg.fit(np.array([[0], [1], [2]]), np.array([0, 1, 2]))
    assert _xgb_n_targets(reg) == 1

    with pytest.raises(TypeError):
        _xgb_n_targets(object())


@pytest.mark.parametrize(
    ['matrix_type', 'value', 'sparse_missing'],
    [(mt, v, sm)
     for mt in [sp.csc_matrix, sp.csr_matrix]
     for v in [0, np.nan, 12]
     for sm in [False, True]
     ])
def test_set_missing_values_to_nan_sparse(matrix_type, value, sparse_missing):
    ms = matrix_type((1, 100))
    ms[0, 54] = 12
    ms[0, 42] = 0
    ms[0, 7] = -13
    m = _missing_values_set_to_nan(ms, value, sparse_missing)
    assert ms[0, 54] == 12
    assert ms[0, 42] == 0
    assert ms[0, 7] == -13
    assert not sp.issparse(m)
    assert m.shape == (100,)
    if sparse_missing:
        assert np.isnan(m[8])
        if value == 0:
            assert np.isnan(m[42])
        else:
            assert m[42] == 0
    else:
        if value == 0:
            assert np.isnan(m[8])
        else:
            assert m[8] == 0
    if value == 0:
        assert np.isnan(m[42])
    elif value == 12:
        assert np.isnan(m[54])
    assert m[7] == -13


def test_parent_value():
    assert _parent_value([{'cover': 10., 'leaf': 15.}]) == 15.
    assert _parent_value([
        {'cover': 10., 'leaf': 15.}, {'cover': 40., 'leaf': 5.}]) == 7.


def test_check_booster_args():
    x, y = np.random.random((10, 2)), np.random.randint(2, size=10)
    regressor = XGBRegressor().fit(x, y)
    classifier = XGBClassifier().fit(x, y)
    booster, is_regression = _check_booster_args(regressor)
    assert is_regression is True
    assert isinstance(booster, xgboost.Booster)
    _, is_regression = _check_booster_args(regressor, is_regression=True)
    assert is_regression is True
    _, is_regression = _check_booster_args(classifier)
    assert is_regression is False
    _, is_regression = _check_booster_args(classifier, is_regression=False)
    assert is_regression is False
    with pytest.raises(ValueError):
        _check_booster_args(classifier, is_regression=True)
    with pytest.raises(ValueError):
        _check_booster_args(regressor, is_regression=False)
    booster = xgboost.Booster()
    _booster, is_regression = _check_booster_args(booster)
    assert _booster is booster
    assert is_regression is None
    _, is_regression = _check_booster_args(booster, is_regression=True)
    assert is_regression is True
