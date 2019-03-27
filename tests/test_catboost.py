from __future__ import absolute_import

import pytest 
import numpy as np 
import catboost
from catboost import CatBoostClassifier,CatBoostRegressor,CatBoost

from .utils import format_as_all

from eli5.catboost import (
    _check_catboost_args
)

from eli5.explain import explain_weights


def test_explain_catboost_catboost(boston_train):
    xs, ys, feature_names = boston_train
    catb = CatBoost().fit(xs,ys)
    res = explain_weights(catb)
    for expl in format_as_all(res,catb):
        assert '12' in expl
    res = explain_weights(catb,feature_names=feature_names)
    for expl in format_as_all(res, catb):
        assert 'LSTAT' in expl

def test_explain_catboost_regressor(boston_train):
    xs, ys, feature_names = boston_train
    catb = CatBoostRegressor().fit(xs,ys)
    assert _check_catboost_args(catb)
    res = explain_weights(catb)
    for expl in format_as_all(res,catb):
        assert '12' in expl
    res = explain_weights(catb,feature_names=feature_names)
    for expl in format_as_all(res, catb):
        assert 'LSTAT' in expl  
    