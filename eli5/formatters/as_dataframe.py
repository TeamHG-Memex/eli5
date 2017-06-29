from itertools import chain
from singledispatch import singledispatch
from typing import Dict, List, Optional
import warnings

import pandas as pd  # type: ignore

from eli5 import explain_weights, explain_prediction
from eli5.base import (
    Explanation, FeatureImportances, TargetExplanation, TransitionFeatureWeights,
)


def explain_weights_df(estimator, **kwargs):
    # type: (...) -> pd.DataFrame
    """ Explain weights and export them to ``pandas.DataFrame``.
    All keyword arguments are passed to :func:`eli5.explain_weights`.
    Weights of all features are exported by default.
    """
    kwargs = _set_defaults(kwargs)
    return format_as_dataframe(explain_weights(estimator, **kwargs))


def explain_weights_dfs(estimator, **kwargs):
    # type: (...) -> Dict[str, pd.DataFrame]
    """ Explain weights and export them to a dict with ``pandas.DataFrame``
    values (as :func:`format_as_dataframes` does).
    All keyword arguments are passed to :func:`eli5.explain_weights`.
    Weights of all features are exported by default.
    """
    kwargs = _set_defaults(kwargs)
    return format_as_dataframes(explain_weights(estimator, **kwargs))


def explain_prediction_df(estimator, doc, **kwargs):
    # type: (...) -> pd.DataFrame
    """ Explain prediction and export explanation to ``pandas.DataFrame``
    All keyword arguments are passed to :func:`eli5.explain_prediction`.
    Weights of all features are exported by default.
    """
    kwargs = _set_defaults(kwargs)
    return format_as_dataframe(explain_prediction(estimator, doc, **kwargs))


def explain_prediction_dfs(estimator, doc, **kwargs):
    # type: (...) -> Dict[str, pd.DataFrame]
    """ Explain prediction and export explanation
    to a dict with ``pandas.DataFrame`` values
    (as :func:`format_as_dataframes` does).
    All keyword arguments are passed to :func:`eli5.explain_prediction`.
    Weights of all features are exported by default.
    """
    kwargs = _set_defaults(kwargs)
    return format_as_dataframes(explain_prediction(estimator, doc, **kwargs))


def _set_defaults(kwargs):
    if 'top' not in kwargs:
        # No limit on number of features by default.
        kwargs['top'] = None
    return kwargs


_EXPORTED_ATTRIBUTES = ['transition_features', 'targets', 'feature_importances']


def format_as_dataframes(explanation):
    # type: (Explanation) -> Dict[str, pd.DataFrame]
    """ Export an explanation to a dictionary with ``pandas.DataFrame`` values
    and string keys that correspond to explanation attributes.
    Use this method if several dataframes can be exported from a single explanation
    (e.g. for CRF explanation with has both feature weights and transition matrix).
    """
    result = {}
    for attr in _EXPORTED_ATTRIBUTES:
        value = getattr(explanation, attr)
        if value:
            result[attr] = format_as_dataframe(value)
    return result


@singledispatch
def format_as_dataframe(explanation):
    # type: (Explanation) -> Optional[pd.DataFrame]
    """ Export an explanation to a single ``pandas.DataFrame``.
    In case several dataframes could be exported by
    :func:`eli5.formatters.as_dataframe.format_as_dataframes`,
    a warning is raised. If no dataframe can be exported, ``None`` is returned.
    This function also accepts some components of the explanation as arguments:
    feature importances, targets, transition features.
    """
    for attr in _EXPORTED_ATTRIBUTES:
        value = getattr(explanation, attr)
        if value:
            other_attrs = [a for a in _EXPORTED_ATTRIBUTES
                           if getattr(explanation, a) and a != attr]
            if other_attrs:
                warnings.warn('Exporting {} to DataFrame, but also {} could be '
                              'exported. Consider using eli5.format_as_dataframes.'
                              .format(attr, ', '.join(other_attrs)))
            return format_as_dataframe(value)


@format_as_dataframe.register(FeatureImportances)
def _feature_importances_to_df(feature_importances):
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
def _targets_to_df(targets):
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
def _transition_features_to_df(transition_features):
    # type: (TransitionFeatureWeights) -> pd.DataFrame
    class_names = list(transition_features.class_names)
    df = pd.DataFrame({
        'from': [f for _ in class_names for f in class_names],
        'to': [f for f in class_names for _ in class_names],
        'coef': transition_features.coef.T.reshape(-1),
    })
    table = pd.pivot_table(df, values='coef', columns=['to'], index=['from'])
    # recover original order
    table = table[class_names]
    table = table.reindex(class_names)
    return table
