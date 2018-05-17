import json

import numpy as np

from eli5.base import (
    Explanation, TargetExplanation, FeatureWeights, FeatureWeight)
from eli5.formatters.as_dict import format_as_dict, _numpy_to_python


# format_as_dict is called in eli5.tests.utils.format_as_all


def test_numpy_to_python():
    x = _numpy_to_python({
        'x': np.int32(12),
        'y': [np.ones(2)],
        'z': {'inner': np.bool_(False)},
    })
    assert x == {
        'x': 12,
        'y': [[1.0, 1.0]],
        'z': {'inner': False},
    }
    json.dumps(x)


def test_format_as_dict():
    assert format_as_dict(Explanation(
        estimator='some estimator',
        targets=[
            TargetExplanation(
                'y', feature_weights=FeatureWeights(
                    pos=[FeatureWeight('a', np.float32(13.0))],
                    neg=[])),
        ],
    )) == {'estimator': 'some estimator',
           'targets': [
               {'target': 'y',
                'feature_weights': {
                    'pos': [{
                        'feature': 'a',
                        'weight': 13.0,
                        'std': None,
                        'value': None}],
                    'pos_remaining': 0,
                    'neg': [],
                    'neg_remaining': 0,
                },
                'score': None,
                'proba': None,
                'weighted_spans': None,
                },
           ],
           'decision_tree': None,
           'description': None,
           'error': None,
           'feature_importances': None,
           'highlight_spaces': None,
           'is_regression': False,
           'method': None,
           'transition_features': None
           }
