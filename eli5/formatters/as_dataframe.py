from itertools import chain
from typing import Any, Dict, List, Optional
import warnings

import pandas as pd

import eli5
from eli5.base import (
    Explanation, FeatureImportances, TargetExplanation,
    TransitionFeatureWeights,
)
from eli5.base_utils import singledispatch


def explain_weights_df(estimator, **kwargs):
    # type: (...) -> pd.DataFrame
    """ Explain weights and export them to ``pandas.DataFrame``.
    All keyword arguments are passed to :func:`eli5.explain_weights`.
    Weights of all features are exported by default.
    """
    kwargs = _set_defaults(kwargs)
    return format_as_dataframe(
        eli5.explain_weights(estimator, **kwargs))


def explain_weights_dfs(estimator, **kwargs):
    # type: (...) -> Dict[str, pd.DataFrame]
    """ Explain weights and export them to a dict with ``pandas.DataFrame``
    values (as :func:`eli5.formatters.as_dataframe.format_as_dataframes` does).
    All keyword arguments are passed to :func:`eli5.explain_weights`.
    Weights of all features are exported by default.
    """
    kwargs = _set_defaults(kwargs)
    return format_as_dataframes(
        eli5.explain_weights(estimator, **kwargs))


def explain_prediction_df(estimator, doc, **kwargs):
    # type: (...) -> pd.DataFrame
    """ Explain prediction and export explanation to ``pandas.DataFrame``
    All keyword arguments are passed to :func:`eli5.explain_prediction`.
    Weights of all features are exported by default.
    """
    kwargs = _set_defaults(kwargs)
    return format_as_dataframe(
        eli5.explain_prediction(estimator, doc, **kwargs))


def explain_prediction_dfs(estimator, doc, **kwargs):
    # type: (...) -> Dict[str, pd.DataFrame]
    """ Explain prediction and export explanation
    to a dict with ``pandas.DataFrame`` values
    (as :func:`eli5.formatters.as_dataframe.format_as_dataframes` does).
    All keyword arguments are passed to :func:`eli5.explain_prediction`.
    Weights of all features are exported by default.
    """
    kwargs = _set_defaults(kwargs)
    return format_as_dataframes(
        eli5.explain_prediction(estimator, doc, **kwargs))


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
    Use this method if several dataframes can be exported from a single
    explanation (e.g. for CRF explanation with has both feature weights
    and transition matrix).
    Note that :func:`eli5.explain_weights` limits number of features
    by default. If you need all features, pass ``top=None`` to
    :func:`eli5.explain_weights`, or use
    :func:`explain_weights_dfs`.
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
    Note that :func:`eli5.explain_weights` limits number of features
    by default. If you need all features, pass ``top=None`` to
    :func:`eli5.explain_weights`, or use
    :func:`explain_weights_df`.
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
    return None


@format_as_dataframe.register(FeatureImportances)
def _feature_importances_to_df(feature_importances):
    # type: (FeatureImportances) -> pd.DataFrame
    weights = feature_importances.importances
    df = pd.DataFrame(
        {'feature': [fw.feature for fw in weights],
         'weight': [fw.weight for fw in weights],
         },
        columns=['feature', 'weight'])
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
    columns = ['target', 'feature', 'weight', 'std', 'value']
    df_data = {f: [] for f in columns}  # type: Dict[str, List[Any]]
    for target in targets:
        assert target.feature_weights is not None
        for fw in chain(target.feature_weights.pos,
                        reversed(target.feature_weights.neg)):
            df_data['target'].append(target.target)
            df_data['feature'].append(fw.feature)
            df_data['weight'].append(fw.weight)
            df_data['std'].append(fw.std)
            df_data['value'].append(fw.value)
    for optional_field in ['std', 'value']:
        if all(x is None for x in df_data[optional_field]):
            df_data.pop(optional_field)
            columns.remove(optional_field)
    return pd.DataFrame(df_data, columns=columns)


@format_as_dataframe.register(TransitionFeatureWeights)
def _transition_features_to_df(transition_features):
    # type: (TransitionFeatureWeights) -> pd.DataFrame
    class_names = list(transition_features.class_names)
    return pd.DataFrame(
        {'from': [f for f in class_names for _ in class_names],
         'to': [f for _ in class_names for f in class_names],
         'coef': transition_features.coef.reshape(-1),
         },
        columns=['from', 'to', 'coef'])
