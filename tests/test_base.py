# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator

from eli5.sklearn import explain_weights_sklearn


def test_repr_html(boston_train):
    X, y, feature_names = boston_train
    reg = LinearRegression()
    reg.fit(X, y)
    res = explain_weights_sklearn(reg)
    html = res._repr_html_()
    assert 'LinearRegression' not in html
    assert 'BIAS' in html


def test_repr_html_error():
    reg = BaseEstimator()
    res = explain_weights_sklearn(reg)
    html = res._repr_html_()
    assert 'BaseEstimator' in html
