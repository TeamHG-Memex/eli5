from __future__ import print_function

import pytest
pd = pytest.importorskip('pandas')
import numpy as np

from eli5 import format_as_dataframes, format_as_dataframe, format_as_text
from eli5.base import (
    Explanation, TargetExplanation, FeatureWeight, FeatureWeights,
    FeatureImportances, TransitionFeatureWeights,
)


def test_targets():
    expl = Explanation(
        estimator='some estimator',
        targets=[
            TargetExplanation(
                'y', feature_weights=FeatureWeights(
                    pos=[FeatureWeight('a', 13),
                         FeatureWeight('b', 5)],
                    neg=[FeatureWeight('neg1', -10),
                         FeatureWeight('neg2', -1)],
                )),
            TargetExplanation(
                'y2', feature_weights=FeatureWeights(
                    pos=[FeatureWeight('f', 1)],
                    neg=[],
                )),
        ],
    )
    df_dict = format_as_dataframes(expl)
    assert isinstance(df_dict, dict)
    assert list(df_dict) == ['targets']
    df = df_dict['targets']
    expected_df = pd.DataFrame(
        {'weight': [13, 5, -1, -10, 1]},
        index=pd.MultiIndex.from_tuples(
            [('y', 'a'), ('y', 'b'), ('y', 'neg2'), ('y', 'neg1'),
             ('y2', 'f')], names=['target', 'feature']))
    print(df, expected_df, sep='\n')
    assert expected_df.equals(df)

    single_df = format_as_dataframe(expl)
    assert expected_df.equals(single_df)


def test_targets_with_value():
    expl = Explanation(
        estimator='some estimator',
        targets=[
            TargetExplanation(
                'y', feature_weights=FeatureWeights(
                    pos=[FeatureWeight('a', 13, value=1),
                         FeatureWeight('b', 5, value=2)],
                    neg=[FeatureWeight('neg1', -10, value=3),
                         FeatureWeight('neg2', -1, value=4)],
                )),
            TargetExplanation(
                'y2', feature_weights=FeatureWeights(
                    pos=[FeatureWeight('f', 1, value=5)],
                    neg=[],
                )),
        ],
    )
    df = format_as_dataframe(expl)
    expected_df = pd.DataFrame(
        {'weight': [13, 5, -1, -10, 1],
         'value': [1, 2, 4, 3, 5]},
        columns=['weight', 'value'],
        index=pd.MultiIndex.from_tuples(
            [('y', 'a'), ('y', 'b'), ('y', 'neg2'), ('y', 'neg1'),
             ('y2', 'f')], names=['target', 'feature']))
    print(df, expected_df, sep='\n')
    assert expected_df.equals(df)


def test_feature_importances():
    expl = Explanation(
        estimator='some estimator',
        feature_importances=FeatureImportances(
            importances=[
                FeatureWeight('a', 1),
                FeatureWeight('b', 2),
            ],
            remaining=10,
        )
    )
    df_dict = format_as_dataframes(expl)
    assert isinstance(df_dict, dict)
    assert list(df_dict) == ['feature_importances']
    df = df_dict['feature_importances']
    expected_df = pd.DataFrame({'weight': [1, 2]}, index=['a', 'b'])
    print(df, expected_df, sep='\n')
    assert expected_df.equals(df)

    single_df = format_as_dataframe(expl)
    assert expected_df.equals(single_df)


def test_feature_importances_std():
    expl = Explanation(
        estimator='some estimator',
        feature_importances=FeatureImportances(
            importances=[
                FeatureWeight('a', 1, std=0.1),
                FeatureWeight('b', 2, std=0.2),
            ],
            remaining=10,
        )
    )
    df = format_as_dataframe(expl)
    expected_df = pd.DataFrame(
        {'weight': [1, 2], 'std': [0.1, 0.2]},
        columns=['weight', 'std'],
        index=['a', 'b'])
    print(df, expected_df, sep='\n')
    assert expected_df.equals(df)


def test_transition_features():
    expl = Explanation(
        estimator='some estimator',
        targets=[
            TargetExplanation(
                'class1', feature_weights=FeatureWeights(
                    pos=[FeatureWeight('pos', 13, value=1)],
                    neg=[],
                )),
            TargetExplanation(
                'class2', feature_weights=FeatureWeights(
                    pos=[FeatureWeight('pos', 13, value=1)],
                    neg=[],
                )),
        ],
        transition_features=TransitionFeatureWeights(
            class_names=['class2', 'class1'],  # reverse on purpose
            coef=np.array([[1.5, 2.5], [3.5, 4.5]]),
        )
    )
    df_dict = format_as_dataframes(expl)
    assert isinstance(df_dict, dict)
    assert set(df_dict) == {'targets', 'transition_features'}
    assert df_dict['targets'].equals(format_as_dataframe(expl.targets))
    df = df_dict['transition_features']
    print(df)
    print(format_as_text(expl))
    assert str(df) == (
        'to      class2  class1\n'
        'from                  \n'
        'class2     1.5     2.5\n'
        'class1     3.5     4.5'
    )

    with pytest.warns(UserWarning):
        single_df = format_as_dataframe(expl)
    assert single_df.equals(df)
