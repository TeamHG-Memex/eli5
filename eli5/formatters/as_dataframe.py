from itertools import chain
from singledispatch import singledispatch
from typing import Dict, List, Optional
import warnings

import pandas as pd  # type: ignore

from eli5.base import (
    Explanation, FeatureImportances, TargetExplanation, TransitionFeatureWeights,
)


EXPORTED_ATTRIBUTES = ['transition_features', 'targets', 'feature_importances']


def format_as_dataframes(expl):
    # type: (Explanation) -> Dict[str, pd.DataFrame]
    """ Export an explanation to a dictionary with pandas.DataFrame values
    and string keys that correspond to explanation attributes.
    Use this method if several dataframes can be exported from a single explanation.
    """
    result = {}
    for attr in EXPORTED_ATTRIBUTES:
        value = getattr(expl, attr)
        if value:
            result[attr] = format_as_dataframe(value)
    return result


@singledispatch
def format_as_dataframe(expl):
    # type: (Explanation) -> Optional[pd.DataFrame]
    """ Export an explanation to a single pandas.DataFrame.
    In case several dataframes could be exported by format_as_dataframes,
    a warning is raised. If no dataframe can be exported, None is returned.
    """
    for attr in EXPORTED_ATTRIBUTES:
        value = getattr(expl, attr)
        if value:
            other_attrs = [a for a in EXPORTED_ATTRIBUTES
                           if getattr(expl, a) and a != attr]
            if other_attrs:
                warnings.warn('Exporting {} to DataFrame, but also {} could be '
                              'exported. Consider using eli5.format_as_dataframes.'
                              .format(attr, ', '.join(other_attrs)))
            return format_as_dataframe(value)


@format_as_dataframe.register(FeatureImportances)
def feature_importances_to_df(feature_importances):
    # type: (FeatureImportances) -> pd.DataFrame
    weights = feature_importances.importances
    df = pd.DataFrame({'weight': [fw.weight for fw in weights]},
                      index=[fw.feature for fw in weights])
    if any(fw.std is not None for fw in weights):
        df['std'] = [fw.std for fw in weights]
    if any(fw.value is not None for fw in weights):
        df['value'] = [fw.value for fw in weights]
    return df


@format_as_dataframe.register(list)
def targets_to_df(targets):
    # type: (List[TargetExplanation]) -> pd.DataFrame
    if targets and not isinstance(targets[0], TargetExplanation):
        raise ValueError('Only lists of TargetExplanation are supported')
    index, weights, stds, values = [], [], [], []
    for target in targets:
        for fw in chain(target.feature_weights.pos,
                        reversed(target.feature_weights.neg)):
            index.append((target.target, fw.feature))
            weights.append(fw.weight)
            stds.append(fw.std)
            values.append(fw.value)
    df = pd.DataFrame(
        {'weight': weights},
        index=pd.MultiIndex.from_tuples(index, names=['target', 'feature']))
    if any(x is not None for x in stds):
        df['std'] = stds
    if any(x is not None for x in values):
        df['value'] = values
    return df


@format_as_dataframe.register(TransitionFeatureWeights)
def transition_features_to_df(transition_features):
    # type: (TransitionFeatureWeights) -> pd.DataFrame
    class_names = transition_features.class_names
    df = pd.DataFrame({
        'from': [f for _ in class_names for f in class_names],
        'to': [f for f in class_names for _ in class_names],
        'coef': transition_features.coef.T.reshape(-1),
    })
    return pd.pivot_table(df, values='coef', columns=['to'], index=['from'])
