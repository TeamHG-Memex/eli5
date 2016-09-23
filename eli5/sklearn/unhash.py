# -*- coding: utf-8 -*-
"""
Utilities to reverse transformation done by FeatureHasher or HashingVectorizer.
"""
from __future__ import absolute_import

from collections import defaultdict
from itertools import chain
from typing import List, Iterable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import HashingVectorizer, FeatureHasher


class InverseFeatureHasher(BaseEstimator):
    def __init__(self, hasher, unkn_template="FEATURE[%d]",
                 signed_feature_names=True):
        # type: (FeatureHasher) -> None
        self.hasher = hasher
        self.n_features = self.hasher.n_features
        self.unkn_template = unkn_template
        self.signed_feature_names = signed_feature_names

    def fit(self, X):
        # type: (Iterable[str]) -> InverseFeatureHasher
        terms = np.array(list(set(X)))
        indices, signs = _get_indices_and_signs(self.hasher, terms)
        self.terms_ = terms
        self.term_columns_ = indices
        self.term_signs_ = signs
        self.collisions_ = _get_collisions(indices)
        self.column_signs_ = self._get_column_signs()
        return self

    def get_feature_names(self):
        return self._get_feature_names()

    def _feature_name_signed(self, i):
        sign = "(+)" if self.term_signs_[i] > 0 else '(-)'
        return sign + self.terms_[i]

    def _feature_name_unsigned(self, i):
        return self.terms_[i]

    def _get_column_signs(self):
        colums_signs = np.zeros(self.n_features, dtype=int)

        for hash_id, term_ids in self.collisions_.items():
            term_signs = self.term_signs_[term_ids]
            if (term_signs < 0).all():
                colums_signs[hash_id] = -1
            elif (term_signs > 0).all():
                colums_signs[hash_id] = 1

        return colums_signs

    def _get_feature_names(self):
        names = np.array(
            [self.unkn_template % d for d in range(self.n_features)],
            dtype=object
        )

        if self.signed_feature_names:
            _get_name = self._feature_name_signed
        else:
            _get_name = self._feature_name_unsigned

        for hash_id, term_ids in self.collisions_.items():
            name = " | ".join(_get_name(i) for i in term_ids)
            names[hash_id] = name
            # todo: better handling of term signs?
            # if len(term_ids) == 1:
            #     name = terms[term_ids[0]]
            # else:
            #     if negated[hash_id] or (term_signs > 0).all():
            #         name = " | ".join(terms[idx] for idx in term_ids)
            #     else:
            #         name = " | ".join(self._feature_name_signed(i)
            #                           for i in term_ids)
        return names

    # def transform(self, X):
    #     # ???
    #     feature_names = self.get_feature_names()
    #     return [
    #         list(feature_names[row.nonzero()[1]])
    #         for row in X
    #     ]


class InverseHashingVectorizer(BaseEstimator):
    def __init__(self, vec):
        # type: (HashingVectorizer) -> None
        self.vec = vec
        self.inverse_hasher = InverseFeatureHasher(vec._get_hasher(),
                                                   signed_feature_names=True)

    def fit(self, X):
        """ Extract possible terms from documents """
        analyze = self.vec.build_analyzer()
        terms = chain.from_iterable(analyze(doc) for doc in X)
        self.inverse_hasher.fit(terms)

    def get_feature_names(self):
        return self.inverse_hasher.get_feature_names()

    # def transform(self, X):
    #     """
    #     Return terms per document with nonzero entries in X.
    #
    #     Parameters
    #     ----------
    #     X : {array, sparse matrix}, shape = [n_samples, n_features]
    #
    #     Returns
    #     -------
    #     X_inv : list of arrays, len = n_samples
    #         List of arrays of terms.
    #     """


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
