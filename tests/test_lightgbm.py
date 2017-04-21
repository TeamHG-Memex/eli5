# -*- coding: utf-8 -*-
from __future__ import absolute_import

import pytest
pytest.importorskip('lightgbm')

from lightgbm import LGBMClassifier, LGBMRegressor


from .test_sklearn_explain_weights import (
    test_explain_tree_classifier as _check_rf_classifier,
    test_explain_random_forest_and_tree_feature_filter as _check_rf_feature_filter,
    test_feature_importances_no_remaining as _check_rf_no_remaining,
    test_explain_tree_regressor as _check_rf_regressor,
)

@pytest.fixture()
def lgb_clf():
    return LGBMClassifier(n_estimators=10,
                          min_child_samples=2,
                          min_child_weight=0)


def test_explain_lightgbm(newsgroups_train, lgb_clf):
    _check_rf_classifier(newsgroups_train, lgb_clf)


def test_explain_lightgbm_feature_filter(newsgroups_train, lgb_clf):
    _check_rf_feature_filter(newsgroups_train, lgb_clf)


def test_feature_importances_no_remaining(lgb_clf):
    _check_rf_no_remaining(lgb_clf)


def test_regressor(boston_train):
    reg = LGBMRegressor()
    _check_rf_regressor(reg, boston_train)