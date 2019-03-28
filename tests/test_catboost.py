from __future__ import absolute_import

import pytest 

pytest.importorskip('catboost')

import numpy as np 
import catboost
from catboost import CatBoostClassifier,CatBoostRegressor,CatBoost,Pool

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
    
def test_explain_catboost_classifier(iris_train):
    x,y,feature_names,target = iris_train
    train = Pool(x,y)
    catb = CatBoostClassifier(iterations=10,
                            learning_rate=1,
                            depth=2,
                            loss_function='MultiClass')
    catb.fit(train)
    res = explain_weights(catb)
    for expl in format_as_all(res,catb):
        assert '3' in expl
    res = explain_weights(catb,feature_names=feature_names)
    for expl in format_as_all(res,catb):
        assert 'petal width (cm)' in expl