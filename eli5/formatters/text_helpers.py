from collections import Counter
import copy
from typing import Callable, List, Optional, Tuple

import numpy as np

from eli5.base import (
    FeatureWeights, FormattedFeatureName, TargetExplanation, WeightedSpans)
from eli5.base_utils import attrs
from .utils import max_or_0


def get_char_weights(weighted_spans, preserve_density=None):
    # type: (WeightedSpans, bool) -> np.ndarray
    """ Return character weights for a text document with highlighted features.
    If preserve_density is True, then color for longer fragments will be
    less intensive than for shorter fragments, so that "sum" of intensities
    will correspond to feature weight.
    If preserve_density is None, then it's value is chosen depending on
    analyzer kind: it is preserved for "char" and "char_wb" analyzers,
    and not preserved for "word" analyzers.
    """
    if preserve_density is None:
        preserve_density = weighted_spans.analyzer.startswith('char')
    char_weights = np.zeros(len(weighted_spans.document))
    feature_counts = Counter(f for f, _, _ in weighted_spans.weighted_spans)
    for feature, spans, weight in weighted_spans.weighted_spans:
        for start, end in spans:
            if preserve_density:
                weight /= (end - start)
            weight /= feature_counts[feature]
            char_weights[start:end] += weight
    return char_weights


@attrs
class PreparedWeightedSpans(object):
    def __init__(self,
                 weighted_spans,  # type: WeightedSpans
                 char_weights,  # type: np.ndarray
                 weight_range,  # type: float
                 ):
        self.weighted_spans = weighted_spans
        self.char_weights = char_weights
        self.weight_range = weight_range


def get_prepared_weighted_spans(targets, preserve_density):
    # type: (List[TargetExplanation], bool) -> List[List[PreparedWeightedSpans]]
    """ Return weighted spans prepared for rendering.
    Calculate a separate weight range for each different weighted
    span (for each different index): each target has the same number
    of weighted spans.
    """
    targets_char_weights = [
        [get_char_weights(ws, preserve_density=preserve_density)
         for ws in t.weighted_spans] if t.weighted_spans else None
        for t in targets]  # type: List[List[np.ndarray]]
    max_idx = max_or_0(len(ch_w or []) for ch_w in targets_char_weights)
    spans_weight_ranges = [
        max_or_0(
            abs(x) for char_weights in targets_char_weights
            for x in char_weights[idx] if char_weights is not None)
        for idx in range(max_idx)]
    return [
        [PreparedWeightedSpans(ws, char_weights, weight_range)
         for ws, char_weights, weight_range in zip(
            t.weighted_spans, t_char_weights, spans_weight_ranges)]
        if t_char_weights is not None else None
        for t, t_char_weights in zip(targets, targets_char_weights)]


def merge_weighted_spans_others(
        target, with_vec_name='{}: {}'.format):
    # type: (TargetExplanation, Callable[[str, str], str]) -> Optional[FeatureWeights]
    """ Merge "others" of a list of weighted spans into a single "others" field.
    with_vec_name is a function that takes vectorizer name and feature name
    to produce the new feature name.
    """
    weighted_spans = target.weighted_spans
    if not weighted_spans:
        return None
    if len(weighted_spans) == 1:
        return weighted_spans[0].other
    return FeatureWeights(
        pos=[_renamed(fw, ws, with_vec_name) for ws in weighted_spans
             for fw in ws.other.pos],
        neg=[_renamed(fw, ws, with_vec_name) for ws in weighted_spans
             for fw in ws.other.neg],
        # All should be the same, so min is fine
        pos_remaining=min(ws.other.pos_remaining for ws in weighted_spans),
        neg_remaining=min(ws.other.neg_remaining for ws in weighted_spans),
    )


def _renamed(fw, ws, with_vec_name):
    if not ws.vec_name:
        return fw
    fw = copy.copy(fw)
    renamed = lambda x: with_vec_name(ws.vec_name, x)
    if isinstance(fw.feature, FormattedFeatureName):
        fw.feature = FormattedFeatureName(renamed(fw.feature.value))
    elif isinstance(fw.feature, list):
        fw.feature = [
            {'name': renamed(x['name']), 'sign': x['sign']} for x in fw.feature]
    else:
        fw.feature = renamed(fw.feature)
    return fw
