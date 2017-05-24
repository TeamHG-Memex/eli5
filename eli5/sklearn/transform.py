"""transform_feature_names implementations for scikit-learn transformers

These are automatically registered for many scikit-learn transformers, but can
be overridden by, for example, registering ``make_tfn_weighted`` with
non-default options for a decomposition transformer (such as PCA).
"""

import itertools
import operator

import numpy as np  # type: ignore
from scipy import sparse  # type: ignore
import six  # type: ignore
from sklearn.pipeline import Pipeline, FeatureUnion  # type: ignore
from sklearn.feature_selection.base import SelectorMixin  # type: ignore
from sklearn.linear_model import (  # type: ignore
    RandomizedLogisticRegression,
    RandomizedLasso,
)
from sklearn.feature_extraction.text import TfidfTransformer  # type: ignore
from sklearn.decomposition import (  # type: ignore
    PCA,
    IncrementalPCA,
    FactorAnalysis,
    FastICA,
    TruncatedSVD,
    NMF,
    SparsePCA,
    MiniBatchSparsePCA,
    SparseCoder,
    DictionaryLearning,
    MiniBatchDictionaryLearning,
    LatentDirichletAllocation)

from eli5.transform import transform_feature_names
from eli5.sklearn.utils import get_feature_names as _get_feature_names


def _attrgetter_or_identity(func):
    if isinstance(func, six.string_types):
        return operator.attrgetter(func)
    return func


class make_tfn_weighted:
    """Makes feature names representing a weighted sum of input features

    An instance can be registered as the handler of ``transform_feature_names``
    for linear transformers.

    Parameters
    ----------
    get_weights : callable or str, optional if top=0
        A function or attribute name which, applied to the registered
        transformer, will return an array of shape (n_outputs, n_inputs)
        or (n_inputs,) for diagonal, describing the linear combination of
        inputs to produce outputs.
    top : int, default=3
        Maximum number of input features to show as contributing to output
        features.
    threshold : float, default 0
        A contributing feature will only be shown if the absolute value of
        its weight exceeds this value.
    show_weight : bool or str, default True
        Whether or not to show the weights of contributing features.
        When a str, is the format of those weights. Defaults to '0.3g'.
    show_idx : bool, default True
        Whether or not to number the output features so that it is easy to
        identify the component.
    func_name : str, optional
        Specifies constant text to prepend to output feature number.
    """
    # TODO: Perhaps have some kind of relative thresholding, either
    #       threshold as a function of the weight matrix, or threshold
    #       relative to highest/immediately bigger contribution to an output
    #       feature.
    def __init__(self, get_weights, top=3, threshold=0,
                 show_weight=True, show_idx=True, func_name=None):
        if not any([top, show_idx, func_name]):
            raise ValueError('At least one of top, show_idx and '
                             'func_name must be set')
        if threshold < 0:
            raise ValueError('Threshold must be >= 0')
        self.main_fmt, self.contrib_fmt, self.contrib_sep = self.build_formats(
            top, show_weight, show_idx, func_name)
        self.get_weights = _attrgetter_or_identity(get_weights)
        self.top = top
        self.threshold = threshold

    @staticmethod
    def build_formats(top, show_weight, show_idx, func_name,
                      contrib_fmt_fmt='{{name}}*{{weight:{}}}',
                      contrib_sep='+'):
        if show_idx and top:
            main_fmt = '{idx:d}:=({feats})'
        elif show_idx:
            main_fmt = '{idx:d}'
        elif top:
            main_fmt = '({feats})'
        else:
            main_fmt = ''
        if func_name:
            escaped_func_name = func_name.replace('{', '{{').replace('}', '}}')
            main_fmt = escaped_func_name + main_fmt

        if not top:
            contrib_fmt = None
            contrib_sep = None
        elif show_weight:
            if show_weight is True:
                show_weight = '0.3g'
            contrib_fmt = contrib_fmt_fmt.format(show_weight)
        else:
            contrib_fmt = '{name}'

        return main_fmt, contrib_fmt, contrib_sep

    @staticmethod
    def find_contributors(W, top, threshold):
        if sparse.issparse(W):
            W = W.tocsr(copy=True)
            W.sum_duplicates()
            W_abs = abs(W)

            def _find_contribs(idx, w, w_abs):
                order = np.argsort(w_abs)[-top:][::-1]
                order = order[w_abs[order] > threshold]
                return idx[order], w[order]

            return (_find_contribs(W.indices[start:stop],
                                   W.data[start:stop],
                                   W_abs.data[start:stop])
                    for start, stop in zip(W.indptr, W.indptr[1:]))

        else:
            W_abs = abs(W)
            # TODO: use argpartition?
            top_idx = np.argsort(W_abs, axis=1)[:, -top:][:, ::-1]
            ix0 = np.arange(W_abs.shape[0]).repeat(top_idx.shape[1])
            ix0 = ix0.reshape(-1, top_idx.shape[1])
            n_contribs = (W_abs[ix0, top_idx] > threshold).sum(axis=1)
            return ((idx[:n], w[:n]) for idx, w, n in
                    zip(top_idx, W[ix0, top_idx], n_contribs))

    def __call__(self, est, in_names=None):
        W = self.get_weights(est)
        if W.ndim == 1:
            # XXX: This implementation is inefficient and could be rewritten
            W = sparse.csr_matrix((W.copy(), np.arange(len(W)),
                                   np.arange(len(W) + 1)))

        in_names = _get_feature_names(est, feature_names=in_names,
                                      num_features=W.shape[1])
        if len(in_names) != W.shape[1]:
            raise ValueError('Got in_names of length {}, but '
                             'weights.shape[1]=={}'.format(len(in_names),
                                                           W.shape[1]))

        if self.top:
            fmt = self.contrib_fmt.format
            contribs = self.find_contributors(W, self.top, self.threshold)
            contribs = list(contribs)
            contrib_sep = self.contrib_sep
            contribs = (contrib_sep.join(fmt(name=in_names[i], weight=w)
                                         for i, w in zip(idx, weights))
                        for idx, weights in contribs)
        else:
            contribs = itertools.repeat('', W.shape[0])

        return [self.main_fmt.format(idx=i, feats=feats)
                for i, feats in enumerate(contribs)]


# Non-trivial scaling:

transform_feature_names.register(TfidfTransformer)(  # type: ignore
    make_tfn_weighted('idf_', func_name='TFIDF', show_idx=False))

# Decomposition (linear weights):

for cls, prefix in [(PCA, 'PCA'), (IncrementalPCA, 'PCA'),
                    (FactorAnalysis, 'FA'), (FastICA, 'ICA'),
                    (TruncatedSVD, 'SVD'), (NMF, 'NMF'),
                    (SparsePCA, 'SPCA'), (MiniBatchSparsePCA, 'SPCA'),
                    (SparseCoder, 'SC'), (DictionaryLearning, 'DL'),
                    (MiniBatchDictionaryLearning, 'DL'),
                    (LatentDirichletAllocation, 'LDA')]:

    transform_feature_names.register(cls)(  # type: ignore
        make_tfn_weighted('components_', func_name=prefix))


# Feature selection:

@transform_feature_names.register(SelectorMixin)
@transform_feature_names.register(RandomizedLogisticRegression)
@transform_feature_names.register(RandomizedLasso)
def _select_names(est, in_names=None):
    mask = est.get_support(indices=False)
    in_names = _get_feature_names(est, feature_names=in_names,
                                  num_features=len(mask))
    return [in_names[i] for i in np.flatnonzero(mask)]


# Pipelines

@transform_feature_names.register(Pipeline)
def _pipeline_names(est, in_names=None):
    names = in_names
    for name, trans in est.steps:
        if trans is not None:
            names = transform_feature_names(trans, names)
    return names


@transform_feature_names.register(FeatureUnion)
def _union_names(est, in_names=None):
    return ['{}:{}'.format(trans_name, feat_name)
            for trans_name, trans, _ in est._iter()
            for feat_name in transform_feature_names(trans, in_names)]
