# -*- coding: utf-8 -*-
"""
Utilities to reverse transformation done by FeatureHasher or HashingVectorizer.
"""
from __future__ import absolute_import
from collections import defaultdict, Counter
from itertools import chain
from typing import List, Iterable, Any, Dict, Tuple, Union

import numpy as np
import six
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    FeatureHasher,
)
from sklearn.pipeline import FeatureUnion

from eli5._feature_names import FeatureNames


class InvertableHashingVectorizer(BaseEstimator, TransformerMixin):
    """
    A wrapper for HashingVectorizer which allows to get meaningful
    feature names. Create it with an existing HashingVectorizer
    instance as an argument::

        vec = InvertableHashingVectorizer(my_hashing_vectorizer)

    Unlike HashingVectorizer it can be fit. During fitting
    :class:`~.InvertableHashingVectorizer` learns which input terms map to
    which feature columns/signs; this allows to provide more meaningful
    :meth:`get_feature_names`. The cost is that it is no longer stateless.

    You can fit :class:`~.InvertableHashingVectorizer` on a random sample
    of documents (not necessarily on the whole training and testing data),
    and use it to inspect an existing HashingVectorizer instance.

    If several features hash to the same value, they are ordered by
    their frequency in documents that were used to fit the vectorizer.

    :meth:`transform` works the same as HashingVectorizer.transform.
    """
    def __init__(self, vec,
                 unkn_template="FEATURE[%d]"):
        # type: (HashingVectorizer, str) -> None
        self.vec = vec
        self.unkn_template = unkn_template
        self.unhasher = FeatureUnhasher(
            hasher=vec._get_hasher(),
            unkn_template=unkn_template,
        )
        self.n_features = vec.n_features  # type: int

    def fit(self, X, y=None):
        """ Extract possible terms from documents """
        self.unhasher.fit(self._get_terms_iter(X))
        return self

    def partial_fit(self, X):
        self.unhasher.partial_fit(self._get_terms_iter(X))
        return self

    def transform(self, X):
        return self.vec.transform(X)

    def get_feature_names(self, always_signed=True):
        # type: (bool) -> FeatureNames
        """
        Return feature names.
        This is a best-effort function which tries to reconstruct feature
        names based on what it has seen so far.

        HashingVectorizer uses a signed hash function. If always_signed is True,
        each term in feature names is prepended with its sign. If it is False,
        signs are only shown in case of possible collisions of different sign.

        You probably want always_signed=True if you're checking
        unprocessed classifier coefficients, and always_signed=False
        if you've taken care of :attr:`column_signs_`.
        """
        return self.unhasher.get_feature_names(
            always_signed=always_signed,
            always_positive=self._always_positive(),
        )

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
        if self._always_positive():
            return np.ones(self.n_features)
        self.unhasher.recalculate_attributes()
        return self.unhasher.column_signs_

    def _always_positive(self):
        # type: () -> bool
        return (
            self.vec.binary
            or getattr(self.vec, 'non_negative', False)
            or not getattr(self.vec, 'alternate_sign', True)
        )


class FeatureUnhasher(BaseEstimator):
    """
    Class for recovering a mapping used by FeatureHasher.
    """
    def __init__(self, hasher, unkn_template="FEATURE[%d]"):
        # type: (FeatureHasher, str) -> None
        if hasher.input_type != 'string':
            raise ValueError("FeatureUnhasher only supports hashers with "
                             "input_type 'string', got %r." % hasher.input_type)
        self.hasher = hasher
        self.n_features = self.hasher.n_features  # type: int
        self.unkn_template = unkn_template
        self._attributes_dirty = True
        self._term_counts = Counter()  # type: Counter

    def fit(self, X, y=None):
        # type: (Iterable[str], Any) -> FeatureUnhasher
        self._term_counts.clear()
        self.partial_fit(X, y)
        self.recalculate_attributes(force=True)
        return self

    def partial_fit(self, X, y=None):
        # type: (Iterable[str], Any) -> FeatureUnhasher
        self._term_counts.update(X)
        self._attributes_dirty = True
        return self

    def get_feature_names(self, always_signed=True, always_positive=False):
        # type: (bool, bool) -> FeatureNames
        self.recalculate_attributes()

        # lists of names with signs of known features
        column_ids, term_names, term_signs = self._get_collision_info()
        feature_names = {}
        for col_id, names, signs in zip(column_ids, term_names, term_signs):
            if always_positive:
                feature_names[col_id] = [{'name': name, 'sign': 1}
                                         for name in names]
            else:
                if not always_signed and _invert_signs(signs):
                    signs = [-sign for sign in signs]
                feature_names[col_id] = [{'name': name, 'sign': sign}
                                         for name, sign in zip(names, signs)]
        return FeatureNames(
            feature_names,
            n_features=self.n_features,
            unkn_template=self.unkn_template)

    def recalculate_attributes(self, force=False):
        # type: (bool) -> None
        """
        Update all computed attributes. It is only needed if you need to access
        computed attributes after :meth:`patrial_fit` was called.
        """
        if not self._attributes_dirty and not force:
            return
        terms = [term for term, _ in self._term_counts.most_common()]
        if six.PY2:
            terms = np.array(terms, dtype=np.object)
        else:
            terms = np.array(terms)
        if len(terms):
            indices, signs = _get_indices_and_signs(self.hasher, terms)
        else:
            indices, signs = np.array([]), np.array([])
        self.terms_ = terms  # type: np.ndarray
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
        # type: () -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]
        column_ids, term_names, term_signs = [], [], []
        for column_id, _term_ids in self.collisions_.items():
            column_ids.append(column_id)
            term_names.append(self.terms_[_term_ids])
            term_signs.append(self.term_signs_[_term_ids])
        return column_ids, term_names, term_signs


def _get_collisions(indices):
    # type: (...) -> Dict[int, List[int]]
    """
    Return a dict ``{column_id: [possible term ids]}``
    with collision information.
    """
    collisions = defaultdict(list)  # type: Dict[int, List[int]]
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


def _invert_signs(signs):
    """ Shall we invert signs?
    Invert if first (most probable) term is negative.
    """
    return signs[0] < 0


def is_invhashing(vec):
    return isinstance(vec, InvertableHashingVectorizer)


def handle_hashing_vec(vec, feature_names, coef_scale, with_coef_scale=True):
    """ Return feature_names and coef_scale (if with_coef_scale is True),
    calling .get_feature_names for invhashing vectorizers.
    """
    needs_coef_scale = with_coef_scale and coef_scale is None
    if is_invhashing(vec):
        if feature_names is None:
            feature_names = vec.get_feature_names(always_signed=False)
        if needs_coef_scale:
            coef_scale = vec.column_signs_
    elif (isinstance(vec, FeatureUnion) and
              any(is_invhashing(v) for _, v in vec.transformer_list) and
              (needs_coef_scale or feature_names is None)):
        _feature_names, _coef_scale = _invhashing_union_feature_names_scale(vec)
        if feature_names is None:
            feature_names = _feature_names
        if needs_coef_scale:
            coef_scale = _coef_scale
    return (feature_names, coef_scale) if with_coef_scale else feature_names


def _invhashing_union_feature_names_scale(vec_union):
    # type: (FeatureUnion) -> Tuple[FeatureNames, np.ndarray]
    feature_names_store = {}  # type: Dict[int, Union[str, List]]
    unkn_template = None
    shift = 0
    coef_scale_values = []
    for vec_name, vec in vec_union.transformer_list:
        if isinstance(vec, InvertableHashingVectorizer):
            vec_feature_names = vec.get_feature_names(always_signed=False)
            unkn_template = vec_feature_names.unkn_template
            for idx, fs in vec_feature_names.feature_names.items():
                new_fs = []
                for f in fs:
                    new_f = dict(f)
                    new_f['name'] = '{}__{}'.format(vec_name, f['name'])
                    new_fs.append(new_f)
                feature_names_store[idx + shift] = new_fs
            coef_scale_values.append((shift, vec.column_signs_))
            shift += vec_feature_names.n_features
        else:
            vec_feature_names = vec.get_feature_names()
            feature_names_store.update(
                (shift + idx, '{}__{}'.format(vec_name, fname))
                for idx, fname in enumerate(vec_feature_names))
            shift += len(vec_feature_names)
    n_features = shift
    feature_names = FeatureNames(
        feature_names=feature_names_store,
        n_features=n_features,
        unkn_template=unkn_template)
    coef_scale = np.ones(n_features) * np.nan
    for idx, values in coef_scale_values:
        coef_scale[idx: idx + len(values)] = values
    return feature_names, coef_scale


def invert_hashing_and_fit(
        vec,  # type: Union[FeatureUnion, HashingVectorizer]
        docs
    ):
    # type: (...) -> Union[FeatureUnion, InvertableHashingVectorizer]
    """ Create an :class:`~.InvertableHashingVectorizer` from hashing
    vectorizer vec and fit it on docs. If vec is a FeatureUnion, do it for all
    hashing vectorizers in the union.
    Return an :class:`~.InvertableHashingVectorizer`, or a FeatureUnion,
    or an unchanged vectorizer.
    """
    if isinstance(vec, HashingVectorizer):
        vec = InvertableHashingVectorizer(vec)
        vec.fit(docs)
    elif (isinstance(vec, FeatureUnion) and
              any(isinstance(v, HashingVectorizer)
                  for _, v in vec.transformer_list)):
        vec = _fit_invhashing_union(vec, docs)
    return vec


def _fit_invhashing_union(vec_union, docs):
    # type: (FeatureUnion, Any) -> FeatureUnion
    """ Fit InvertableHashingVectorizer on doc inside a FeatureUnion.
    """
    return FeatureUnion(
        [(name, invert_hashing_and_fit(v, docs))
         for name, v in vec_union.transformer_list],
        transformer_weights=vec_union.transformer_weights,
        n_jobs=vec_union.n_jobs)
