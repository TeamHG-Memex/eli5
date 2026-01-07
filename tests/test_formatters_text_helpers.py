import numpy as np

from eli5.base import (
    TargetExplanation, WeightedSpans, DocWeightedSpans, FeatureWeights)
from eli5.formatters.text_helpers import (
    PreparedWeightedSpans, prepare_weighted_spans)


def test_prepare_weighted_spans():
    targets = [
        TargetExplanation(
            target='one',
            feature_weights=FeatureWeights(pos=[], neg=[]),
            weighted_spans=WeightedSpans(
                docs_weighted_spans=[
                    DocWeightedSpans(
                        document='ab',
                        spans=[
                            ('a', [(0, 1)], 1.5),
                            ('b', [(1, 2)], 2.5),
                        ],
                    ),
                    DocWeightedSpans(
                        document='xy',
                        spans=[
                            ('xy', [(0, 2)], -4.5),
                        ],
                    )]
            )),
        TargetExplanation(
            target='two',
            feature_weights=FeatureWeights(pos=[], neg=[]),
            weighted_spans=WeightedSpans(
                docs_weighted_spans=[
                    DocWeightedSpans(
                        document='abc',
                        spans=[
                            ('a', [(0, 1)], 0.5),
                            ('c', [(2, 3)], 3.5),
                        ],
                    ),
                    DocWeightedSpans(
                        document='xz',
                        spans=[
                            # char_wb at the start of the document
                            (' xz', [(-1, 2)], 1.5),
                        ],
                    )],
            )),
    ]
    assert prepare_weighted_spans(targets, preserve_density=False) == [
        [
            PreparedWeightedSpans(
                targets[0].weighted_spans.docs_weighted_spans[0],
                char_weights=np.array([1.5, 2.5]),
                weight_range=3.5),
            PreparedWeightedSpans(
                targets[0].weighted_spans.docs_weighted_spans[1],
                char_weights=np.array([-4.5, -4.5]),
                weight_range=4.5),
        ],
        [
            PreparedWeightedSpans(
                targets[1].weighted_spans.docs_weighted_spans[0],
                char_weights=np.array([0.5, 0, 3.5]),
                weight_range=3.5),
            PreparedWeightedSpans(
                targets[1].weighted_spans.docs_weighted_spans[1],
                char_weights=np.array([1.5, 1.5]),
                weight_range=4.5),
        ],
    ]
