# -*- coding: utf-8 -*-
from __future__ import absolute_import
import pytest
pytest.importorskip('xgboost')
from xgboost import XGBClassifier, XGBRegressor

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
