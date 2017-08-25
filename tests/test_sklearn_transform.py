import re

import pytest
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectPercentile,
    SelectKBest,
    SelectFpr,  # TODO: add tests and document
    SelectFdr,  # TODO: add tests and document
    SelectFwe,  # TODO: add tests and document
    GenericUnivariateSelect,
    VarianceThreshold,
    RFE,
    RFECV,
    SelectFromModel,
)
from sklearn.linear_model import (
    LogisticRegression,
    RandomizedLogisticRegression,
    RandomizedLasso,  # TODO: add tests and document
)
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler,
)
from sklearn.pipeline import FeatureUnion, make_pipeline

from eli5 import transform_feature_names
from eli5.sklearn import PermutationImportance


class MyFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, :3]

    def get_feature_names(self):
        return ['f1', 'f2', 'f3']


def selection_score_func(X, y):
    return np.array([1, 2, 3, 4])


@pytest.mark.parametrize('transformer,expected', [
    (MyFeatureExtractor(), ['f1', 'f2', 'f3']),

    (make_pipeline(StandardScaler(), MyFeatureExtractor()),
     ['f1', 'f2', 'f3']),
    (make_pipeline(MinMaxScaler(), MyFeatureExtractor()),
     ['f1', 'f2', 'f3']),
    (make_pipeline(MaxAbsScaler(), MyFeatureExtractor()),
     ['f1', 'f2', 'f3']),
    (make_pipeline(RobustScaler(), MyFeatureExtractor()),
     ['f1', 'f2', 'f3']),
    (StandardScaler(), ['<NAME0>', '<NAME1>', '<NAME2>', '<NAME3>']),
    (MinMaxScaler(), ['<NAME0>', '<NAME1>', '<NAME2>', '<NAME3>']),
    (MaxAbsScaler(), ['<NAME0>', '<NAME1>', '<NAME2>', '<NAME3>']),
    (RobustScaler(), ['<NAME0>', '<NAME1>', '<NAME2>', '<NAME3>']),

    (SelectKBest(selection_score_func, k=1),
     ['<NAME3>']),
    (SelectKBest(selection_score_func, k=2),
     ['<NAME2>', '<NAME3>']),
    (FeatureUnion([('k', SelectKBest(selection_score_func, k=2)),
                   ('p', SelectPercentile(selection_score_func, 30))]),
     ['k:<NAME2>', 'k:<NAME3>', 'p:<NAME3>']),
    (VarianceThreshold(0.0), ['<NAME0>', '<NAME1>', '<NAME2>', '<NAME3>']),
    (VarianceThreshold(1.0), ['<NAME2>']),
    (GenericUnivariateSelect(), ['<NAME2>']),
    (GenericUnivariateSelect(mode='k_best', param=2), ['<NAME2>', '<NAME3>']),
    (SelectFromModel(LogisticRegression('l1', C=0.01, random_state=42)),
     ['<NAME0>', '<NAME2>']),
    (SelectFromModel(
        PermutationImportance(
            LogisticRegression(random_state=42),
            cv=5, random_state=42, refit=False,
        ),
        threshold=0.1,
     ),
     ['<NAME2>', '<NAME3>']),
    (RFE(LogisticRegression(random_state=42), 2),
     ['<NAME1>', '<NAME3>']),
    (RFECV(LogisticRegression(random_state=42)),
     ['<NAME0>', '<NAME1>', '<NAME2>', '<NAME3>']),
    (RandomizedLogisticRegression(random_state=42),
     ['<NAME1>', '<NAME2>', '<NAME3>']),
])
def test_transform_feature_names_iris(transformer, expected, iris_train):
    X, y, _, _ = iris_train
    transformer.fit(X, y)
    # Test in_names being provided
    res = transform_feature_names(
        transformer, ['<NAME0>', '<NAME1>', '<NAME2>', '<NAME3>']
    )
    assert res == expected
    # Test in_names being None
    expected_default_names = [re.sub('<NAME([0-9]+)>', r'x\1', name)
                              for name in expected]
    assert transform_feature_names(transformer, None) == expected_default_names
