import re

import pytest
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.datasets import load_iris
from eli5 import transform_feature_names

iris_X, iris_y = load_iris(return_X_y=True)


class MyFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, :3]

    def get_feature_names(self):
        return ['f1', 'f2', 'f3']


def selection_score_func(X, y):
    return np.array([1, 2, 3])


@pytest.mark.parametrize('transformer,expected', [
    (MyFeatureExtractor(), ['f1', 'f2', 'f3']),
    (SelectKBest(selection_score_func, k=1),
     ['<NAME2>']),
    (SelectKBest(selection_score_func, k=2),
     ['<NAME1>', '<NAME2>']),
    (FeatureUnion([('k', SelectKBest(selection_score_func, k=2)),
                   ('p', SelectPercentile(selection_score_func, 40))]),
     ['k:<NAME1>', 'k:<NAME2>', 'p:<NAME2>']),
])
def test_transform_feature_names_iris(transformer, expected):
    transformer.fit(iris_X, iris_y)
    # Test in_names being provided
    assert (transform_feature_names(transformer,
                                    ['<NAME0>', '<NAME1>', '<NAME2>'])
            == expected)
    # Test in_names being None
    expected_default_names = [re.sub('<NAME([0-9]+)>', r'x\1', name)
                              for name in expected]
    assert transform_feature_names(transformer, None) == expected_default_names
