# -*- coding: utf-8 -*-
from __future__ import absolute_import

import pytest
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
pytest.importorskip('xgboost')
from xgboost import XGBClassifier, XGBRegressor

from eli5.base import TargetExplanation
from eli5.xgboost import _parse_tree_dump, _xgb_n_targets
from eli5.explain import explain_prediction, explain_weights
from eli5.formatters.text import format_as_text
from eli5.formatters import fields
from .utils import format_as_all, get_all_features
from .test_sklearn_explain_weights import (
    test_explain_random_forest as _check_rf,
    test_explain_random_forest_and_tree_feature_re as _check_rf_feature_re,
    test_feature_importances_no_remaining as _check_rf_no_remaining,
)


def test_explain_xgboost(newsgroups_train):
    _check_rf(newsgroups_train, XGBClassifier())


def test_explain_xgboost_feature_re(newsgroups_train):
    _check_rf_feature_re(newsgroups_train, XGBClassifier())


def test_feature_importances_no_remaining():
    _check_rf_no_remaining(XGBClassifier())


def test_explain_xgboost_regressor(boston_train):
    xs, ys, feature_names = boston_train
    reg = XGBRegressor()
    reg.fit(xs, ys)
    res = explain_weights(reg)
    for expl in  format_as_all(res, reg):
        assert 'x12' in expl
    res = explain_weights(reg, feature_names=feature_names)
    for expl in format_as_all(res, reg):
        assert 'LSTAT' in expl


def test_explain_prediction_clf_binary(newsgroups_train_binary_big):
    docs, ys, target_names = newsgroups_train_binary_big
    vec = CountVectorizer(stop_words='english')
    clf = XGBClassifier(n_estimators=100, max_depth=2, missing=0)
    xs = vec.fit_transform(docs)
    clf.fit(xs, ys)
    res = explain_prediction(
        clf, 'computer graphics in space: a sign of atheism',
        vec=vec, target_names=target_names)
    format_as_all(res, clf)
    _check_scores(res)
    weights = res.targets[0].feature_weights
    pos_features = get_all_features(weights.pos)
    neg_features = get_all_features(weights.neg)
    assert 'graphics' in pos_features
    assert 'computer' in pos_features
    assert 'atheism' in neg_features


def test_explain_prediction_clf_multitarget(newsgroups_train):
    docs, ys, target_names = newsgroups_train
    vec = CountVectorizer(stop_words='english')
    xs = vec.fit_transform(docs)
    clf = XGBClassifier(n_estimators=100, max_depth=2, missing=0)
    clf.fit(xs, ys)
    res = explain_prediction(
        clf, 'computer graphics in space: a new religion',
        vec=vec, target_names=target_names)
    format_as_all(res, clf)
    _check_scores(res)
    graphics_weights = res.targets[1].feature_weights
    assert 'computer' in get_all_features(graphics_weights.pos)
    religion_weights = res.targets[3].feature_weights
    assert 'religion' in get_all_features(religion_weights.pos)


def test_explain_prediction_clf_xor():
    true_xs = [[np.random.randint(2), np.random.randint(2)] for _ in range(100)]
    xs = np.array([[np.random.normal(x, 0.2), np.random.normal(y, 0.2)]
                   for x, y in true_xs])
    ys = np.array([x == y for x, y in true_xs])
    clf = XGBClassifier(n_estimators=100, max_depth=2)
    clf.fit(xs, ys)
    for x in [[0, 1], [1, 0], [0, 0], [1, 1]]:
        res = explain_prediction(clf, np.array(x))
        print(x)
        print(format_as_text(res, show=fields.WEIGHTS))
        _check_scores(res)


def test_explain_prediction_clf_interval():
    true_xs = [[np.random.randint(3), np.random.randint(10)] for _ in range(1000)]
    xs = np.array([[np.random.normal(x, 0.2), np.random.normal(y, 0.2)]
                   for x, y in true_xs])
    ys = np.array([x == 1 for x, _ in true_xs])
    clf = XGBClassifier(n_estimators=100, max_depth=2)
    clf.fit(xs, ys)
    for x in [[0, 1], [1, 1], [2, 1], [0.8, 5], [1.2, 5]]:
        res = explain_prediction(clf, np.array(x))
        print(x)
        print(format_as_text(res, show=fields.WEIGHTS))
        _check_scores(res)


def test_explain_prediction_reg(boston_train):
    xs, ys, feature_names = boston_train
    reg = XGBRegressor()
    reg.fit(xs, ys)
    res = explain_prediction(reg, xs[0])
    _check_scores(res)
    for expl in  format_as_all(res, reg):
        assert 'x12' in expl
    res = explain_prediction(reg, xs[0], feature_names=feature_names)
    _check_scores(res)
    for expl in format_as_all(res, reg):
        assert 'LSTAT' in expl


def _check_scores(explanation):
    for target in explanation.targets:
        _check_target_score(target)


def _check_target_score(target):
    # type: (TargetExplanation) -> None
    weights = target.feature_weights
    # else the check is invalid
    assert weights.neg_remaining == weights.pos_remaining == 0
    assert np.isclose(
        target.score,
        sum(fw.weight for fw in weights.pos) +
        sum(fw.weight for fw in weights.neg))


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
