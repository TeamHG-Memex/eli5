import re
import pickle

import pytest
import numpy as np
from scipy import sparse
from hypothesis import given, example, assume, settings as hyp_settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectPercentile,
    SelectKBest,
    SelectFpr,  # TODO: add tests and document
    SelectFdr,  # TODO: add tests and document
    SelectFwe,  # TODO: add tests and document
    GenericUnivariateSelect,
    VarianceThreshold,
    RFE,
    RFECV,
    SelectFromModel,
)
from sklearn.linear_model import (
    LogisticRegression,
    RandomizedLogisticRegression,
    RandomizedLasso,  # TODO: add tests and document
)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion
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


from eli5 import transform_feature_names
from eli5.sklearn.transform import make_tfn_weighted


class MyFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, :3]

    def get_feature_names(self):
        return ['f1', 'f2', 'f3']


def selection_score_func(X, y):
    return np.arange(X.shape[1])


@pytest.mark.parametrize('transformer,expected', [
    (MyFeatureExtractor(), ['f1', 'f2', 'f3']),
    (SelectKBest(selection_score_func, k=1), ['x3']),
    (SelectKBest(selection_score_func, k=2), ['x2', 'x3']),
    (VarianceThreshold(0.0), ['x0', 'x1', 'x2', 'x3']),
    (VarianceThreshold(1.0), ['x2']),
    (GenericUnivariateSelect(), ['x2']),
    (GenericUnivariateSelect(mode='k_best', param=2), ['x2', 'x3']),
    (SelectFromModel(LogisticRegression('l1', C=0.01, random_state=42)),
     ['x0', 'x2']),
    (RFE(LogisticRegression(random_state=42), 2), ['x1', 'x3']),
    (RFECV(LogisticRegression(random_state=42)), ['x0', 'x1', 'x2', 'x3']),
    (RandomizedLogisticRegression(random_state=42), ['x1', 'x2', 'x3']),
    (FeatureUnion([('k', SelectKBest(selection_score_func, k=2)),
                   ('p', SelectPercentile(selection_score_func, 25))]),
     ['k:x2', 'k:x3', 'p:x3']),
    # Decompositions with 2 components each weighting 3 features
    (PCA(n_components=2),
     [r'PCA0:=\((x[0-9]\*.+){3}\)', r'PCA1:=\((x[0-9]\*.+){3}\)']),
    (IncrementalPCA(n_components=2),
     [r'PCA0:=\((x[0-9]\*.+){3}\)', r'PCA1:=\((x[0-9]\*.+){3}\)']),
    (FactorAnalysis(n_components=2),
     [r'FA0:=\((x[0-9]\*.+){3}\)', r'FA1:=\((x[0-9]\*.+){3}\)']),
    (FastICA(n_components=2),
     [r'ICA0:=\((x[0-9]\*.+){3}\)', r'ICA1:=\((x[0-9]\*.+){3}\)']),
    (TruncatedSVD(n_components=2),
     [r'SVD0:=\((x[0-9]\*.+){3}\)', r'SVD1:=\((x[0-9]\*.+){3}\)']),
    (NMF(n_components=2),
     [r'NMF0:=\((x[0-9]\*.+){3}\)', r'NMF1:=\((x[0-9]\*.+){3}\)']),
    (SparsePCA(n_components=2),
     [r'SPCA0:=\((x[0-9]\*.+){3}\)', r'SPCA1:=\((x[0-9]\*.+){3}\)']),
    (MiniBatchSparsePCA(n_components=2),
     [r'SPCA0:=\((x[0-9]\*.+){3}\)', r'SPCA1:=\((x[0-9]\*.+){3}\)']),
    (SparseCoder(dictionary=np.array([[1, 2, 3, 4], [5, 6, 7, 8]])),
     [r'SC0:=\((x[0-9]\*.+){3}\)', r'SC1:=\((x[0-9]\*.+){3}\)']),
    (DictionaryLearning(n_components=2),
     [r'DL0:=\((x[0-9]\*.+){3}\)', r'DL1:=\((x[0-9]\*.+){3}\)']),
    (MiniBatchDictionaryLearning(n_components=2),
     [r'DL0:=\((x[0-9]\*.+){3}\)', r'DL1:=\((x[0-9]\*.+){3}\)']),
    (LatentDirichletAllocation(n_topics=2),
     [r'LDA0:=\((x[0-9]\*.+){3}\)', r'LDA1:=\((x[0-9]\*.+){3}\)']),
    (TfidfTransformer(),
     [r'TFIDF\(x0\*.+\)', r'TFIDF\(x1\*.+\)', r'TFIDF\(x2\*.+\)',
      r'TFIDF\(x3\*.+\)', ]),
])
def test_transform_feature_names_match(transformer, expected, iris_train):
    X, y, _, _ = iris_train
    transformer.fit(X, y)
    actual = transform_feature_names(transformer)
    assert len(actual) == len(expected)
    for expected_name, actual_name in zip(expected, actual):
        assert re.match(expected_name, actual_name)


@pytest.mark.parametrize('transformer', [
    SelectKBest(k=2),
    FeatureUnion([('k', SelectKBest(k=2)),
                  ('p', SelectPercentile(percentile=40))]),
    LatentDirichletAllocation(),
    PCA(),
    TruncatedSVD(),
    TfidfTransformer(),
])
def test_transform_feature_names_in_names(transformer, iris_train):
    X, y, _, _ = iris_train
    transformer.fit(X, y)
    specified = transform_feature_names(
        transformer,
        ['<NAME0>', '<NAME1>', '<NAME2>', '<NAME3>'])
    # ensure that the subtitution below does something
    assert any('<NAME' in name for name in specified)
    expected_default_names = [re.sub('<NAME([0-9]+)>', r'x\1', name)
                              for name in specified]
    assert transform_feature_names(transformer, None) == expected_default_names

    for n_in_names in [3, 5]:
        with pytest.raises(ValueError) as exc_info:
            transform_feature_names(transformer, in_names=['x'] * n_in_names)
        assert 'feature_names has a wrong length' in str(exc_info.value)


class Namespace:
    """Provides attributes otherwise supplied by a fitted transformer"""

    def __init__(self, **d):
        self.__dict__.update(d)


@st.composite
def float_formats(draw):
    alignment = draw(st.sampled_from(['', '<', '>', '^']))
    padding = draw(st.sampled_from(['', '-', '0']))
    width = draw(st.one_of(st.just(''), st.integers(0, 100)))
    decimals = str(draw(st.one_of(st.just(''), st.integers(0, 100))))
    if decimals:
        decimals = '.' + decimals
    float_code = draw(st.sampled_from('fFgGeE'))
    return '%s%s%s%s%s' % (alignment, padding, width, decimals, float_code)


@given(W=np_st.arrays(np_st.floating_dtypes(),
                      np_st.array_shapes(min_dims=2, max_dims=2)),
       top=st.integers(min_value=0, max_value=15),
       threshold=st.floats(min_value=0),
       func_name=st.text(),
       show_idx=st.booleans(),
       show_weight=st.one_of(st.booleans(), float_formats()),
       get_weights=st.sampled_from(['weights_', lambda t: t.weights_]),
       )
def test_make_tfn_weighted(get_weights, W, top, threshold, func_name, show_idx,
                           show_weight):
    assume(np.all(np.isfinite(W)))
    assume(any([top, func_name, show_idx]))
    # Could mock find_contributors() and use fake formats
    tfn = make_tfn_weighted(get_weights=get_weights, top=top,
                            threshold=threshold,
                            func_name=func_name, show_idx=show_idx,
                            show_weight=show_weight)

    fmts = tfn.build_formats(top=top, show_weight=show_weight,
                             show_idx=show_idx, func_name=func_name)
    main_fmt, contrib_fmt, contrib_sep = fmts
    contrib_sep = contrib_sep or ''
    contrib_fmt = contrib_fmt or ''

    est = Namespace(weights_=W)
    names = tfn(est)
    assert len(names) == W.shape[0]

    for i, contribs in enumerate(tfn.find_contributors(W, top, threshold)):
        feats = contrib_sep.join(contrib_fmt.format(name='x' + str(j),
                                                    weight=w)
                                 for j, w in zip(*contribs))
        assert main_fmt.format(idx=i, feats=feats) == names[i]


@given(w=np_st.arrays(np_st.floating_dtypes(),
                      np_st.array_shapes(min_dims=1, max_dims=1)),
       top=st.integers(min_value=0, max_value=15),
       threshold=st.floats(min_value=0),
       func_name=st.text(),
       show_idx=st.booleans(),
       show_weight=st.one_of(st.booleans(), float_formats()),
       get_weights=st.sampled_from(['weights_', lambda t: t.weights_]),
       )
@hyp_settings(max_examples=30)
def test_make_tfn_weighted_1d(get_weights, w, top, threshold, func_name,
                              show_idx, show_weight):
    assume(np.all(np.isfinite(w)))
    assume(any([top, func_name, show_idx]))
    # Could mock find_contributors() and use fake formats
    tfn = make_tfn_weighted(get_weights=get_weights, top=top,
                            threshold=threshold,
                            func_name=func_name, show_idx=show_idx,
                            show_weight=show_weight)
    assert (tfn(Namespace(weights_=np.diagflat(w))) ==
            tfn(Namespace(weights_=w)))


def test_make_tfn_weighted_invalid_threshold():
    with pytest.raises(ValueError) as exc_info:
        make_tfn_weighted('weights_', threshold=-1)
    assert 'Threshold must be >= 0' in str(exc_info.value)


def test_make_tfn_weighted_nothing_to_show():
    with pytest.raises(ValueError) as exc_info:
        make_tfn_weighted('weights_', top=0, show_idx=False)
    assert 'At least one' in str(exc_info.value)


@pytest.mark.parametrize(
    'top,show_weight,show_idx,main_fmt,contrib_fmt,contrib_sep', [
        (0, False, False, 'X', None, None),
        (0, False, True, 'X{idx:d}', None, None),
        (0, True, False, 'X', None, None),
        (0, True, True, 'X{idx:d}', None, None),
        (1, False, False, 'X({feats})', '{name}', '+'),
        (1, False, True, 'X{idx:d}:=({feats})', '{name}', '+'),
        (1, True, False, 'X({feats})', '{name}*{weight:0.3g}', '+'),
        (1, True, True, 'X{idx:d}:=({feats})', '{name}*{weight:0.3g}', '+'),
        (1, '.5f', False, 'X({feats})', '{name}*{weight:.5f}', '+'),
        (1, '.5f', True, 'X{idx:d}:=({feats})', '{name}*{weight:.5f}', '+'),
        (5, True, False, 'X({feats})', '{name}*{weight:0.3g}', '+'),
        (5, True, True, 'X{idx:d}:=({feats})', '{name}*{weight:0.3g}', '+'),
        (5, '.5E', False, 'X({feats})', '{name}*{weight:.5E}', '+'),
        (5, '.5E', True, 'X{idx:d}:=({feats})', '{name}*{weight:.5E}', '+'),
    ])
@pytest.mark.parametrize('func_name,exp_func_name', [
    ('', ''),
    (None, ''),
    ('blah', 'blah'),
    ('bl{}ah', 'bl{{}}ah'),
    ('bl{ah', 'bl{{ah'),
])
def test_make_tfn_weighted_build_formats(top, show_weight, show_idx, func_name,
                                         main_fmt, contrib_fmt, contrib_sep,
                                         exp_func_name):
    if not (top or show_idx or func_name):
        # this case tested in test_make_tfn_weighted_invalid
        return
    fmts = make_tfn_weighted.build_formats(top=top, show_weight=show_weight,
                                           show_idx=show_idx,
                                           func_name=func_name)
    assert fmts == (main_fmt.replace('X', exp_func_name),
                    contrib_fmt, contrib_sep)


W_ALL_BINARY = [
    [1, 1, 1],
    [1, 1, 2],
    [1, 2, 1],
    [1, 2, 2],
    [2, 1, 1],
    [2, 1, 2],
    [2, 2, 2]
]


@pytest.mark.parametrize('convert', [
    np.array, sparse.csr_matrix, sparse.dia_matrix
])
@given(W=np_st.arrays(st.one_of(np_st.floating_dtypes(),
                                np_st.integer_dtypes()),
                      np_st.array_shapes(min_dims=2, max_dims=2)),
       top=st.integers(min_value=1, max_value=15),
       threshold=st.floats(min_value=0))
@example(W=W_ALL_BINARY, top=1, threshold=0)  # tests tie-breaking
@example(W=W_ALL_BINARY, top=3, threshold=1)  # tests threshold boundary
def test_make_tfn_weighted_find_contributors(W, convert, top, threshold):
    assume(np.all(np.isfinite(W)))
    W = np.array(W)
    W_conv = convert(W)
    W_conv_pkl = pickle.dumps(W_conv)
    contribs = list(make_tfn_weighted.find_contributors(W_conv, top,
                                                        threshold))
    assert W_conv_pkl == pickle.dumps(W_conv)  # ensure W_conv not modified

    all_idx, all_weights = zip(*contribs)
    assert len(all_idx) == W.shape[0]
    for i, (idx, weights) in enumerate(zip(all_idx, all_weights)):
        abs_weights = np.abs(weights)
        # check weights match idx
        assert np.all(W[i, idx] == weights)
        # check top is satisfied
        assert len(idx) <= min(top, W.shape[1])
        # check idx all distinct
        assert len(np.unique(idx)) == len(idx)
        # check threshold is satisfied
        if len(weights) > 0:
            assert min(abs_weights) > threshold
            # check nothing is bigger in remainder
            rem = np.delete(np.arange(W.shape[1]), idx)
            if len(rem) > 0:
                assert min(abs_weights) >= max(abs(W[i, rem]))
                # if could not fill all top slots, remainder is strictly less
                if len(idx) < min(top, W.shape[1]):
                    assert min(abs_weights) > max(abs(W[i, rem]))
        # check weights are descending
        assert np.all(np.diff(abs_weights) <= 1e-15)
        # TODO: could check ties are broken deterministically


def test_nested_pipelines():
    # TODO
    pass
