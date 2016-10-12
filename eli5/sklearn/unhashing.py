# -*- coding: utf-8 -*-
"""
Utilities to reverse transformation done by FeatureHasher or HashingVectorizer.
"""
from __future__ import absolute_import

from functools import partial
from collections import defaultdict, Counter
from itertools import chain
from typing import List, Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer, FeatureHasher


class InvertableHashingVectorizer(BaseEstimator, TransformerMixin):
    """
    A wrapper for HashingVectorizer which allows to get meaningful
    feature names. Create it with an existing HashingVectorizer
    instance as an argument::

        vec = InvertableHashingVectorizer(my_hashing_vectorizer)

    Unlike HashingVectorizer it can be fit. During fitting
    InvertableHashingVectorizer learns which input terms map to which
    feature columns/signs; this allows to provide more meaningful
    :meth:`get_feature_names`. The cost is that it is no longer stateless.

    You can fit InvertableHashingVectorizer on a random sample of documents
    (not necessarily on the whole training and testing data), and use it
    to inspect an existing HashingVectorizer instance.

    If several features hash to the same value, they are ordered by
    their frequency in documents that were used to fit the vectorizer.

    :meth:`transform` works the same as HashingVectorizer.transform.
    """
    def __init__(self, vec,
                 unkn_template="FEATURE[%d]"):
        # type: (HashingVectorizer, str, bool) -> None
        self.vec = vec
        self.unkn_template = unkn_template
        self.unhasher = FeatureUnhasher(
            hasher=vec._get_hasher(),
            unkn_template=unkn_template,
        )
        self.n_features = vec.n_features

    def fit(self, X, y=None):
        """ Extract possible terms from documents """
        self.unhasher.fit(self._get_terms_iter(X))
        return self

    def partial_fit(self, X):
        self.unhasher.partial_fit(self._get_terms_iter(X))
        return self

    def transform(self, X, y=None):
        return self.vec.transform(X, y)

    def get_feature_names(self, always_signed=True):
        """
        Return feature names.
        This is a best-effort function which tries to reconstruct feature
        names based on what it have seen so far.

        HashingVectorizer uses a signed hash function. If always_signed is True,
        each term in feature names is prepended with its sign. If it is False,
        signs are only shown in case of possible collisions of different sign.

        You probably want always_signed=True if you're checking
        unprocessed classifier coefficients, and always_signed=False
        if you've taken care of :attr:`column_signs_`.
        """
        return self.unhasher.get_feature_names(always_signed=always_signed)

    def _get_terms_iter(self, X):
        analyze = self.vec.build_analyzer()
        return chain.from_iterable(analyze(doc) for doc in X)

    @property
    def column_signs_(self):
        """
        Return a numpy array with expected signs of features.
        Values are

        * +1 when all known terms which map to the column have positive sign;
        * -1 when all known terms which map to the column have negative sign;
        * ``nan`` when there are both positive and negative known terms
          for this column, or when there is no known term which maps to this
          column.
        """
        self.unhasher.recalculate_attributes()
        return self.unhasher.column_signs_


class FeatureUnhasher(BaseEstimator):
    """
    Class for recovering a mapping used by FeatureHasher.
    """
    def __init__(self, hasher, unkn_template="FEATURE[%d]"):
        # type: (FeatureHasher) -> None
        if hasher.input_type != 'string':
            raise ValueError("FeatureUnhasher only supports hashers with "
                             "input_type 'string', got %r." % hasher.input_type)
        self.hasher = hasher
        self.n_features = self.hasher.n_features
        self.unkn_template = unkn_template
        self._attributes_dirty = True
        self._term_counts = Counter()

    def fit(self, X, y=None):
        # type: (Iterable[str], None) -> FeatureUnhasher
        self._term_counts.clear()
        self.partial_fit(X, y)
        self.recalculate_attributes(force=True)
        return self

    def partial_fit(self, X, y=None):
        # type: (Iterable[str], None) -> FeatureUnhasher
        self._term_counts.update(X)
        self._attributes_dirty = True
        return self

    def get_feature_names(self, always_signed=True):
        self.recalculate_attributes()

        # names of unknown features
        names = np.array(
            [self.unkn_template % i for i in range(self.n_features)],
            dtype=object
        )

        # names of known features
        column_ids, term_names, term_signs = self._get_collision_info()
        _fmt = partial(_format_name, always_signed=always_signed)
        names_formatted = list(map(_fmt, term_names, term_signs))
        names[column_ids] = names_formatted
        return names

    def recalculate_attributes(self, force=False):
        """
        Update all computed attributes. It is only needed if you need to access
        computed attributes after :meth:`patrial_fit` was called.
        """
        if not self._attributes_dirty and not force:
            return
        terms = np.array([term for term, _ in self._term_counts.most_common()])
        if len(terms):
            indices, signs = _get_indices_and_signs(self.hasher, terms)
        else:
            indices, signs = np.array([]), np.array([])
        self.terms_ = terms
        self.term_columns_ = indices
        self.term_signs_ = signs
        self.collisions_ = _get_collisions(indices)
        self.column_signs_ = self._get_column_signs()
        self._attributes_dirty = False

    def _get_column_signs(self):
        colums_signs = np.ones(self.n_features) * np.nan
        for hash_id, term_ids in self.collisions_.items():
            term_signs = self.term_signs_[term_ids]
            if _invert_signs(term_signs):
                colums_signs[hash_id] = -1
            elif (term_signs > 0).all():
                colums_signs[hash_id] = 1
        return colums_signs

    def _get_collision_info(self):
        column_ids, term_names, term_signs = [], [], []
        for column_id, _term_ids in self.collisions_.items():
            column_ids.append(column_id)
            term_names.append(self.terms_[_term_ids])
            term_signs.append(self.term_signs_[_term_ids])
        return column_ids, term_names, term_signs


def _get_collisions(indices):
    """
    Return a dict ``{column_id: [possible term ids]}``
    with collision information.
    """
    collisions = defaultdict(list)
    for term_id, hash_id in enumerate(indices):
        collisions[hash_id].append(term_id)
    return dict(collisions)


def _get_indices_and_signs(hasher, terms):
    """
    For each term from ``terms`` return its column index and sign,
    as assigned by FeatureHasher ``hasher``.
    """
    X = _transform_terms(hasher, terms)
    indices = X.nonzero()[1]
    signs = X.sum(axis=1).A.ravel()
    return indices, signs


def _transform_terms(hasher, terms):
    return hasher.transform(np.array(terms).reshape(-1, 1))


def _signed(name, sign):
    """
    >>> _signed("foo", +1)
    'foo'
    >>> _signed("foo", -1)
    '(-)foo'
    """
    txt = "" if sign > 0 else '(-)'
    return "".join([txt, name])


def _format_name(names, signs, sep=" | ", always_signed=False):
    r"""
    Format feature name for hashed features.
    If always_signed is False (default), sign is only added if it is ambiguous.

    >>> _format_name(["foo"], [-1])
    'foo'
    >>> _format_name(["foo"], [-1], always_signed=True)
    '(-)foo'
    >>> _format_name(["foo"], [+1])
    'foo'
    >>> _format_name(["foo", "bar"], [-1, -1])
    'foo | bar'
    >>> _format_name(["foo", "bar"], [-1, +1])
    'foo | (-)bar'
    >>> _format_name(["foo", "bar"], [1, -1])
    'foo | (-)bar'
    """
    if not always_signed and _invert_signs(signs):
        signs = [-sign for sign in signs]
    return sep.join(_signed(n, s) for n, s in zip(names, signs))


def _invert_signs(signs):
    """ Shall we invert signs?
    Invert if first (most probable) term is negative.
    """
    return signs[0] < 0


def handle_hashing_vec(vec, feature_names, coef_scale):
    if is_invhashing(vec):
        if feature_names is None:
            feature_names = vec.get_feature_names(always_signed=False)
        if coef_scale is None:
            coef_scale = vec.column_signs_
    return feature_names, coef_scale


def is_invhashing(vec):
    return isinstance(vec, InvertableHashingVectorizer)
