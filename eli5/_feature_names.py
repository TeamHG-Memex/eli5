import re
import six
from typing import Any, Iterable, Tuple, Sized, List

import numpy as np  # type: ignore


class FeatureNames(Sized):
    """
    A list-like object with feature names. It allows
    feature names for unknown features to be generated using
    a provided template, and to avoid making copies of large objects
    in get_feature_names.
    """
    def __init__(self, feature_names=None, bias_name=None,
                 unkn_template=None, n_features=None):
        if not (feature_names is not None or
                    (unkn_template is not None and n_features)):
            raise ValueError(
                'Pass feature_names or unkn_template and n_features')
        if feature_names is not None:
            if not isinstance(feature_names, (list, dict, np.ndarray)):
                raise TypeError('Unexpected feature_names type')
            if n_features is not None and n_features != len(feature_names):
                if not isinstance(feature_names, dict):
                    raise ValueError(
                        'n_features should match feature_names length')
                elif unkn_template is None:
                    raise ValueError(
                        'unkn_template should be set for sparse features')
        self.feature_names = feature_names
        self._own_feature_names = False  # can not modify them while it's False
        self.unkn_template = unkn_template
        self.n_features = n_features or len(feature_names)
        self.bias_name = bias_name

    def __repr__(self):
        return '<FeatureNames: {} features {} bias>'.format(
            self.n_features, 'with' if self.has_bias else 'without')

    def __len__(self):
        return self.n_features + int(self.has_bias)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._slice(idx)
        if isinstance(idx, np.ndarray):
            return [self[i] for i in idx]
        if self.has_bias and idx == self.bias_idx:
            return self.bias_name
        if 0 <= idx < self.n_features:
            try:
                return self.feature_names[idx]
            except (TypeError, KeyError, IndexError):
                return self.unkn_template % idx
        raise IndexError('Feature index out of range')

    def _slice(self, aslice):
        # type: (slice) -> Any
        if isinstance(self.feature_names, (list, np.ndarray)):
            # Fast path without going through __getitem__
            if self.has_bias:
                lst = list(self.feature_names)
                lst.append(self.bias_name)
            else:
                lst = self.feature_names
            return lst[aslice]
        else:
            indices = range(len(self))[aslice]
            return [self[idx] for idx in indices]

    @property
    def has_bias(self):
        return self.bias_name is not None

    @property
    def bias_idx(self):
        if self.has_bias:
            return self.n_features

    def filtered_by_re(self, feature_re):
        # type: (str) -> Tuple[FeatureNames, List[int]]
        """ Return feature names filtered by a regular expression ``feature_re``,
        and indices of filtered elements.
        """
        indices = []
        filtered_feature_names = []
        indexed_names = None  # type: Iterable[Tuple[int, Any]]
        if isinstance(self.feature_names, (np.ndarray, list)):
            indexed_names = enumerate(self.feature_names)
        elif isinstance(self.feature_names, dict):
            indexed_names = six.iteritems(self.feature_names)
        elif self.feature_names is None:
            indexed_names = []
        assert indexed_names is not None
        if isinstance(feature_re, six.string_types):
            filter_fn = lambda x: re.search(feature_re, x, re.U)
        else:
            filter_fn = lambda x: re.search(feature_re, x)
        for idx, name in indexed_names:
            if any(filter_fn(n) for n in _feature_names(name)):
                indices.append(idx)
                filtered_feature_names.append(name)
        if self.has_bias and filter_fn(self.bias_name):
            indices.append(self.bias_idx)
        return (
            FeatureNames(
                filtered_feature_names,
                bias_name=self.bias_name,
                unkn_template=self.unkn_template,
            ),
            indices)

    def add_feature(self, feature):
        # type: (Any) -> int
        """ Add a new feature name, return it's index.
        """
        # A copy of self.feature_names is always made, because it might be
        # "owned" by someone else.
        # It's possible to make the copy only at the first call to
        # self.add_feature to improve performance.
        idx = self.n_features
        if isinstance(self.feature_names, (list, np.ndarray)):
            if (not self._own_feature_names or
                    isinstance(self.feature_names, np.ndarray)):
                self.feature_names = list(self.feature_names)
            self.feature_names.append(feature)
        elif isinstance(self.feature_names, dict):
            if not self._own_feature_names:
                self.feature_names = dict(self.feature_names)
            self.feature_names[idx] = feature
        elif self.feature_names is None:
            self.feature_names = {idx: feature}
        self.n_features += 1
        self._own_feature_names = True
        return idx


def _feature_names(name):
    if isinstance(name, bytes):
        return [name.decode('utf8')]
    elif isinstance(name, list):
        return [x['name'] for x in name]
    else:
        return [name]

