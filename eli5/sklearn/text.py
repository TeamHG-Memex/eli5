from __future__ import absolute_import
from typing import Any, Union, Callable, Dict, List, Optional, Set, Tuple

from sklearn.pipeline import FeatureUnion
try:
    from sklearn.feature_extraction.text import _VectorizerMixin as VectorizerMixin
except ImportError:  # Changed in scikit-learn 0.22
    from sklearn.feature_extraction.text import VectorizerMixin

from eli5.base import (
    DocWeightedSpans, WeightedSpans, FeatureWeights, FeatureWeight,
    TargetExplanation)
from eli5.formatters import FormattedFeatureName
from eli5.sklearn.unhashing import InvertableHashingVectorizer
from eli5.sklearn._span_analyzers import build_span_analyzer


def get_weighted_spans(doc, vec, feature_weights):
    # type: (Any, Any, FeatureWeights) -> Optional[WeightedSpans]
    """ If possible, return a dict with preprocessed document and a list
    of spans with weights, corresponding to features in the document.
    """
    if isinstance(vec, FeatureUnion):
        return _get_weighted_spans_from_union(doc, vec, feature_weights)
    else:
        result = _get_doc_weighted_spans(doc, vec, feature_weights)
        if result is not None:
            found_features, doc_weighted_spans = result
            return WeightedSpans(
                [doc_weighted_spans],
                other=_get_other(feature_weights, [('', found_features)]),
            )
    return None


def add_weighted_spans(doc, vec, vectorized, target_expl):
    # type: (Any, Any, bool, TargetExplanation) -> None
    """
    Compute and set ``target_expl.weighted_spans`` attribute, when possible.
    """
    if vec is None or vectorized:
        return

    assert target_expl.feature_weights is not None
    weighted_spans = get_weighted_spans(doc, vec, target_expl.feature_weights)
    if weighted_spans:
        target_expl.weighted_spans = weighted_spans


FoundFeatures = Dict[Tuple[str, int], float]


def _get_doc_weighted_spans(doc,
                            vec,
                            feature_weights,  # type: FeatureWeights
                            feature_fn=None   # type: Optional[Callable[[str], str]]
                            ):
    # type: (...) -> Optional[Tuple[FoundFeatures, DocWeightedSpans]]
    if isinstance(vec, InvertableHashingVectorizer):
        vec = vec.vec

    if hasattr(vec, 'get_doc_weighted_spans'):
        return vec.get_doc_weighted_spans(doc, feature_weights, feature_fn)

    if not isinstance(vec, VectorizerMixin):
        return None

    span_analyzer, preprocessed_doc = build_span_analyzer(doc, vec)
    if span_analyzer is None:
        return None

    feature_weights_dict = _get_feature_weights_dict(feature_weights,
                                                     feature_fn)
    spans = []
    found_features = {}
    for f_spans, feature in span_analyzer(preprocessed_doc):
        if feature not in feature_weights_dict:
            continue
        weight, key = feature_weights_dict[feature]
        spans.append((feature, f_spans, weight))
        # XXX: this assumes feature names are unique
        found_features[key] = weight

    return found_features, DocWeightedSpans(
        document=preprocessed_doc,
        spans=spans,
        preserve_density=vec.analyzer.startswith('char'),
    )


def _get_feature_weights_dict(feature_weights,  # type: FeatureWeights
                              feature_fn        # type: Optional[Callable[[str], str]]
                              ):
    # type: (...) -> Dict[str, Tuple[float, Tuple[str, int]]]
    """ Return {feat_name: (weight, (group, idx))} mapping. """
    return {
        # (group, idx) is an unique feature identifier, e.g. ('pos', 2)
        feat_name: (fw.weight, (group, idx))
        for group in ['pos', 'neg']
        for idx, fw in enumerate(getattr(feature_weights, group))
        for feat_name in _get_features(fw.feature, feature_fn)
    }


def _get_features(feature, feature_fn=None):
    if isinstance(feature, list):
        features = [f['name'] for f in feature]
    else:
        features = [feature]
    if feature_fn:
        features = list(filter(None, map(feature_fn, features)))
    return features


def _get_weighted_spans_from_union(doc, vec_union, feature_weights):
    # type: (Any, FeatureUnion, FeatureWeights) -> Optional[WeightedSpans]
    docs_weighted_spans = []
    named_found_features = []
    for vec_name, vec in vec_union.transformer_list:
        vec_prefix = '{}__'.format(vec_name)

        def feature_fn(name):
            if isinstance(name, FormattedFeatureName):
                return
            if not name.startswith(vec_prefix):
                return  # drop feature
            return name[len(vec_prefix):]  # remove prefix

        result = _get_doc_weighted_spans(doc, vec, feature_weights, feature_fn)
        if result:
            found_features, doc_weighted_spans = result
            doc_weighted_spans.vec_name = vec_name
            named_found_features.append((vec_name, found_features))
            docs_weighted_spans.append(doc_weighted_spans)

    if docs_weighted_spans:
        return WeightedSpans(
            docs_weighted_spans,
            other=_get_other(feature_weights, named_found_features),
        )
    else:
        return None


def _get_other(feature_weights, named_found_features):
    # type: (FeatureWeights, List[Tuple[str, FoundFeatures]]) -> FeatureWeights
    # search for items that were not accounted at all.
    other_items = []  # type: List[FeatureWeight]
    accounted_keys = set()  # type: Set[Tuple[str, int]]
    all_found_features = set()  # type: Set[Tuple[str, int]]
    for _, found_features in named_found_features:
        all_found_features.update(found_features)

    for group in ['pos', 'neg']:
        for idx, fw in enumerate(getattr(feature_weights, group)):
            key = (group, idx)
            if key not in all_found_features and key not in accounted_keys:
                other_items.append(fw)
                accounted_keys.add(key)

    for vec_name, found_features in named_found_features:
        if found_features:
            other_items.append(FeatureWeight(
                feature=FormattedFeatureName(
                    '{}Highlighted in text (sum)'.format(
                        '{}: '.format(vec_name) if vec_name else '')),
                weight=sum(found_features.values())))

    other_items.sort(key=lambda x: abs(x.weight), reverse=True)
    return FeatureWeights(
        pos=[fw for fw in other_items if fw.weight >= 0],
        neg=[fw for fw in other_items if fw.weight < 0],
        pos_remaining=feature_weights.pos_remaining,
        neg_remaining=feature_weights.neg_remaining,
    )
