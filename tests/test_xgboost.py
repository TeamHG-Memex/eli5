# -*- coding: utf-8 -*-
from __future__ import absolute_import
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import CountVectorizer
pytest.importorskip('xgboost')
from xgboost import XGBClassifier, XGBRegressor

from eli5.xgboost import parse_tree_dump
from eli5.explain import explain_prediction
from .utils import format_as_all
from .test_sklearn_explain_weights import (
    test_explain_random_forest as _check_rf,
    test_explain_random_forest_and_tree_feature_re as _check_rf_feature_re,
    test_feature_importances_no_remaining as _check_rf_no_remaining,
)

# TODO: XGBRegressor


def test_explain_xgboost(newsgroups_train):
    _check_rf(newsgroups_train, XGBClassifier())


def test_explain_xgboost_feature_re(newsgroups_train):
    _check_rf_feature_re(newsgroups_train, XGBClassifier())


def test_feature_importances_no_remaining():
    _check_rf_no_remaining(XGBClassifier())


def test_explain_prediction_clf_binary():
    xs, ys = make_classification(n_features=4, n_informative=3, n_redundant=1)
    clf = XGBClassifier(n_estimators=5, max_depth=3)
    clf.fit(xs, ys)
    res = explain_prediction(clf, xs[0])
    format_as_all(res, clf)


def test_explain_prediction_clf_multitarget(newsgroups_train):
    docs, y, target_names = newsgroups_train
    vec = CountVectorizer()
    xs = vec.fit_transform(docs)
    clf = XGBClassifier()
    clf.fit(xs, y)
    res = explain_prediction(clf, docs[0], vec=vec, target_names=target_names)
    format_as_all(res, clf)


def test_parse_tree_dump():
    text_dump = '''\
0:[f2<-0.391121] yes=1,no=2,missing=1
	1:[f3<-1.15681] yes=3,no=4,missing=3
		3:leaf=0.0545455
		4:leaf=0.180952
	2:[f3<1.7511] yes=5,no=6,missing=5
		5:[f2<-0.0136578] yes=7,no=8,missing=7
			7:leaf=-0.0545455
			8:leaf=-0.182979
		6:leaf=0.111111
'''
    assert parse_tree_dump(text_dump) == {
        'children': [
            {'children': [{'leaf': 0.0545455, 'nodeid': 3},
                          {'leaf': 0.180952, 'nodeid': 4}],
             'depth': 1,
             'missing': 3,
             'no': 4,
             'nodeid': 1,
             'split': 'f3',
             'split_condition': -1.15681,
             'yes': 3},
            {'children': [
                {'children': [{'leaf': -0.0545455, 'nodeid': 7},
                              {'leaf': -0.182979, 'nodeid': 8}],
                 'depth': 2,
                 'missing': 7,
                 'no': 8,
                 'nodeid': 5,
                 'split': 'f2',
                 'split_condition': -0.0136578,
                 'yes': 7},
                {'leaf': 0.111111, 'nodeid': 6}],
             'depth': 1,
             'missing': 5,
             'no': 6,
             'nodeid': 2,
             'split': 'f3',
             'split_condition': 1.7511,
             'yes': 5}],
        'depth': 0,
        'missing': 1,
        'no': 2,
        'nodeid': 0,
        'split': 'f2',
        'split_condition': -0.391121,
        'yes': 1}
