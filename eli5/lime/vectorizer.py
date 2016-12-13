# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from eli5.base import DocWeightedSpans
from eli5.sklearn.text import _get_feature_weights_dict
from .textutils import TokenizedText, DEFAULT_TOKEN_PATTERN


class SingleDocumentVectorizer(BaseEstimator, TransformerMixin):
    """ Fake vectorizer which converts document just to a vector of ones """

    def __init__(self, token_pattern=DEFAULT_TOKEN_PATTERN):
        self.token_pattern = token_pattern

    def fit(self, X, y=None):
        self.doc_ = X[0]
        self.text_ = TokenizedText(self.doc_)
        return self

    def transform(self, X):
        # assert X[0] == self.doc_
        return np.ones(len(self.text_.tokens)).reshape((1, -1))

    def get_doc_weighted_spans(self, doc, feature_weights, feature_fn):
        feature_weights_dict = _get_feature_weights_dict(feature_weights,
                                                         feature_fn)
        spans = []
        found_features = {}
        for idx, (span, feature) in enumerate(self.text_.spans_and_tokens):
            featname = self._featname(idx, feature)
            try:
                weight, key = feature_weights_dict[featname]
            except KeyError:
                pass
            else:
                spans.append((feature, [span], weight))
                # XXX: this assumes feature names are unique
                found_features[key] = weight

        doc_weighted_spans = DocWeightedSpans(
            document=doc,
            spans=spans,
            preserve_density=False,
        )
        return found_features, doc_weighted_spans

    def _featname(self, idx, token):
        return "[{}] {}".format(idx, token)

    def get_feature_names(self):
        return [self._featname(idx, token)
                for idx, token in enumerate(self.text_.tokens)]
