from itertools import chain
from typing import List, Optional

import pandas as pd

from eli5.base import Explanation, FeatureImportances, TargetExplanation


def format_as_dataframe(expl):
    # type: (Explanation) -> Optional[pd.DataFrame]
    """ Export an explanation to pandas.DataFrame. Only target weights and
    feature importances can be exported, else None is returned.
    """
    if expl.targets:
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
