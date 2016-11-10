import re

import numpy as np


class FeatureNames(object):
    """
    A list-like object with feature names. It allows
    feature names for unknown features to be generated using
    a provided template, and to avoid making copies of large objects
    in get_feature_names.
    """
    def __init__(self, feature_names=None, bias_name=None,
                 unkn_template=None, n_features=None):
        assert (feature_names is not None or
                (unkn_template is not None and n_features))
        self.feature_names = feature_names
        self.unkn_template = unkn_template
        self.n_features = n_features or len(feature_names)
        self.bias_name = bias_name

    def __repr__(self):
        return '<FeatureNames: {} features {} bias>'.format(
            self.n_features, 'with' if self.has_bias else 'without')

    def __len__(self):
        return self.n_features + int(self.has_bias)

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray):
            return [self[i] for i in idx]
        n = len(self)
        if self.has_bias and idx == n - 1:
            return self.bias_name
        if 0 <= idx < n:
            try:
                return self.feature_names[idx]
            except (TypeError, KeyError, IndexError):
                if self.unkn_template is None:
                    raise IndexError('Feature index out of range')
                return self.unkn_template % idx
        raise IndexError('Feature index out of range')

    @property
    def has_bias(self):
        return self.bias_name is not None

    def filtered_by_re(self, feature_re):
        """ Return feature names filtered by a regular expression ``feature_re``,
        and indices of filtered elements.
        """
        indices = []
        filtered_feature_names = []
        for idx, name in enumerate(self):
            if any(re.search(feature_re, n) for n in _feature_names(name)):
                indices.append(idx)
                filtered_feature_names.append(name)
        return (
            FeatureNames(
                filtered_feature_names,
                bias_name=self.bias_name,
                unkn_template=self.unkn_template,
            ),
            indices)


def _feature_names(name):
    if isinstance(name, bytes):
        return [name.decode('utf8')]
    elif isinstance(name, list):
        return [x['name'] for x in name]
    else:
        return [name]

