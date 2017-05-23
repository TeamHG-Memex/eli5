from itertools import chain
from typing import List, Optional

import pandas as pd

from eli5.base import (
    Explanation, FeatureImportances, TargetExplanation, TransitionFeatureWeights,
)


def format_as_dataframe(expl):
    # type: (Explanation) -> Optional[pd.DataFrame]
    """ Export an explanation to pandas.DataFrame. Only target weights and
    feature importances can be exported, else None is returned.
    """
    if expl.transition_features:
        return transition_features_to_df(expl.transition_features)
    elif expl.targets:
        return targets_to_df(expl.targets)
    elif expl.feature_importances:
        return feature_importances_to_df(expl.feature_importances)


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


def targets_to_df(targets):
    # type: (List[TargetExplanation]) -> pd.DataFrame
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


def transition_features_to_df(transition_features):
    # type: (TransitionFeatureWeights) -> pd.DataFrame
    class_names = transition_features.class_names
    df = pd.DataFrame({
        'from': [f for _ in class_names for f in class_names],
        'to': [f for f in class_names for _ in class_names],
        'coef': transition_features.coef.reshape(-1),
    })
    return pd.pivot_table(df, values='coef', columns=['to'], index=['from'])
