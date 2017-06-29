from __future__ import print_function
from itertools import chain

import pytest
pd = pytest.importorskip('pandas')
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor

from eli5 import (
    format_as_dataframes, format_as_dataframe,
    explain_weights_df, explain_weights_dfs,
    explain_prediction_df, explain_prediction_dfs,
    format_as_text, explain_weights, explain_prediction,
)
from eli5.base import (
    Explanation, TargetExplanation, FeatureWeight, FeatureWeights,
    FeatureImportances, TransitionFeatureWeights,
)


def test_explain_weights(boston_train):
    X, y, feature_names = boston_train
    reg = LinearRegression()
    reg.fit(X, y)
    expl = explain_weights(reg)
    df = format_as_dataframe(expl)
    check_targets_dataframe(df, expl)
    check_targets_dataframe(explain_weights_df(reg), expl)
    df_dict = explain_weights_dfs(reg)
    assert set(df_dict.keys()) == {'targets'}
    check_targets_dataframe(df_dict['targets'], expl)


def check_targets_dataframe(df, expl):
    assert list(df.columns) == ['weight']
    for target in expl.targets:
        feature_weights = target.feature_weights
        for fw in chain(feature_weights.pos, feature_weights.neg):
            assert df.loc[target.target, fw.feature]['weight'] == fw.weight


def test_explain_weights_fi(boston_train):
    X, y, feature_names = boston_train
    reg = ExtraTreesRegressor()
    reg.fit(X, y)
    expl = explain_weights(reg)
    df = format_as_dataframe(expl)
    assert list(df.columns) == ['weight', 'std']
    for fw in expl.feature_importances.importances:
        df_fw = df.loc[fw.feature]
        assert np.isclose(df_fw['weight'], fw.weight)
        assert np.isclose(df_fw['std'], fw.std)


def test_explain_prediction(boston_train):
    X, y, feature_names = boston_train
    reg = LinearRegression()
    reg.fit(X, y)
    expl = explain_prediction(reg, X[0])
    df = format_as_dataframe(expl)
    check_prediction_df(df, expl)
    check_prediction_df(explain_prediction_df(reg, X[0]), expl)
    df_dict = explain_prediction_dfs(reg, X[0])
    assert set(df_dict.keys()) == {'targets'}
    check_prediction_df(df_dict['targets'], expl)


def check_prediction_df(df, expl):
    assert list(df.columns) == ['weight', 'value']
    target = expl.targets[0].target
    feature_weights = expl.targets[0].feature_weights
    for fw in chain(feature_weights.pos, feature_weights.neg):
        df_fw = df.loc[target, fw.feature]
        assert df_fw['weight'] == fw.weight
        assert df_fw['value'] == fw.value


@pytest.mark.parametrize(
    ['with_std', 'with_value'],
    [[False, False], [True, False], [False, True]])
def test_targets(with_std, with_value):
    expl = Explanation(
        estimator='some estimator',
        targets=[
            TargetExplanation(
                'y', feature_weights=FeatureWeights(
                    pos=[FeatureWeight('a', 13,
                                       std=0.13 if with_std else None,
                                       value=2 if with_value else None),
                         FeatureWeight('b', 5,
                                       std=0.5 if with_std else None,
                                       value=1 if with_value else None)],
                    neg=[FeatureWeight('neg1', -10,
                                       std=0.2 if with_std else None,
                                       value=5 if with_value else None),
                         FeatureWeight('neg2', -1,
                                       std=0.3 if with_std else None,
                                       value=4 if with_value else None)],
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
    if with_std:
        expected_df['std'] = [0.13, 0.5, 0.3, 0.2, None]
    if with_value:
        expected_df['value'] = [2, 1, 4, 5, None]
    print(df, expected_df, sep='\n')
    assert expected_df.equals(df)

    single_df = format_as_dataframe(expl)
    assert expected_df.equals(single_df)


def test_bad_list():
    with pytest.raises(ValueError):
        format_as_dataframe([1])


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


@pytest.mark.parametrize(
    ['with_std', 'with_value'],
    [[False, False], [True, False], [False, True]])
def test_feature_importances(with_std, with_value):
    expl = Explanation(
        estimator='some estimator',
        feature_importances=FeatureImportances(
            importances=[
                FeatureWeight('a', 1,
                              std=0.1 if with_std else None,
                              value=1 if with_value else None),
                FeatureWeight('b', 2,
                              std=0.2 if with_std else None,
                              value=3 if with_value else None),
            ],
            remaining=10,
        )
    )
    df_dict = format_as_dataframes(expl)
    assert isinstance(df_dict, dict)
    assert list(df_dict) == ['feature_importances']
    df = df_dict['feature_importances']
    expected_df = pd.DataFrame({'weight': [1, 2]}, index=['a', 'b'])
    if with_std:
        expected_df['std'] = [0.1, 0.2]
    if with_value:
        expected_df['value'] = [1, 3]
    print(df, expected_df, sep='\n')
    assert expected_df.equals(df)

    single_df = format_as_dataframe(expl)
    assert expected_df.equals(single_df)


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
