"""transform_feature_names implementations for scikit-learn transformers
"""

import itertools
import operator

import six  # type: ignore
import numpy as np  # type: ignore
from sklearn.pipeline import Pipeline, FeatureUnion  # type: ignore
from sklearn.feature_selection.base import SelectorMixin  # type: ignore
from sklearn.feature_extraction.text import TfidfTransformer  # type: ignore
from sklearn.decomposition import (  # type: ignore
    LatentDirichletAllocation,
    TruncatedSVD,
    PCA)
from sklearn.preprocessing import (  # type: ignore
    Imputer,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    Normalizer,
    Binarizer,
    PolynomialFeatures)
from sklearn.ensemble import RandomTreesEmbedding

from eli5.transform import transform_feature_names
from eli5.sklearn.utils import get_feature_names as _get_feature_names


# XXX Should these generic things be in eli5.transform?
# I've left them here to make use of eli5.sklearn.utils.get_feature_names


def _attrgetter_or_identity(func):
    if isinstance(func, six.string_types):
        return operator.attrgetter(func)
    return func


def _formatted_names(fmt, get_n_inputs=None):
    get_n_inputs = _attrgetter_or_identity(get_n_inputs)

    def transform_names(est, in_names=None):
        if get_n_inputs is not None:
            in_names = _get_feature_names(est, feature_names=in_names,
                                          num_features=get_n_inputs(est))
        return [fmt.format(name) for name in in_names]
    return transform_names


def _component_names(fmt, get_n_outputs):
    get_n_outputs = _attrgetter_or_identity(get_n_outputs)

    def transform_names(est, in_names=None):
        return [fmt.format(i) for i in range(get_n_outputs(est))]
    return transform_names


def make_tfn_featurewise(func_name, get_n_inputs=None):
    """Makes feature names of form func(input_name)

    Parameters
    ----------
    func_name : str
        Specifies constant text to prepend to input feature name.
    get_n_inputs : callable or str, optional
        A function or attribute name which, applied to the registered
        transformer, will return the number of features input (equivalently,
        output) by it.  Only required when in_names is not provided to
        transform_feature_names.

    Returns
    -------
    tfn : callable
        A transform_feature_names implementation
    """
    return _formatted_names(func_name + '({})', get_n_inputs)


def make_tfn_components(func_name, get_n_outputs):
    """Makes feature names of form func0, func1, ...

    Parameters
    ----------
    func_name : str
        Specifies constant text to prepend to output feature number.
    get_n_outputs : callable or str
        A function or attribute name which, applied to the registered
        transformer, will return the number of features output by it.

    Returns
    -------
    tfn : callable
        A transform_feature_names implementation
    """
    return _component_names(func_name + '{}', get_n_outputs)


def make_tfn_weighted(get_weights=None, max_features=3,
                      threshold=None, abs=False,
                      show_weights=True,
                      func_name=None):
    """Makes feature names representing a weighted sum of input features

    Parameters
    ----------
    get_weights : callable or str, optional if max_features=0
        A function or attribute name which, applied to the registered
        transformer, will return an array of shape (n_outputs, n_inputs)
        describing the linear combination of inputs to produce outputs.
    func_name : str, optional
        Specifies constant text to prepend to output feature number.

    Returns
    -------
    tfn : callable
        A transform_feature_names implementation
    """

    if max_features and get_weights is None:
        raise ValueError('Require get_weights if max_features != 0')

    get_weights = _attrgetter_or_identity(get_weights)

    if threshold is None:
        threshold = 0. if abs else -np.inf

    if show_weights:
        if show_weights is True:
            show_weights = '0.2g'
        feat_fmt = '{weight:%s}*{name}' % show_weights
    else:
        feat_fmt = '{name}'
    feat_join = '+'

    if max_features and func_name is not None:
        fmt = '{func_name}{i}=({feats})'
    elif max_features:
        fmt = '({feats})'
    else:
        fmt = '{func_name}{i}'

    def transform_names(est, in_names=None):
        weights = orig_weights = get_weights(est)
        in_names = _get_feature_names(est, feature_names=in_names,
                                      num_features=weights.shape[1])
        if max_features:
            # TODO: support sparse weights
            if abs:
                weights = np.abs(weights)
            order = np.argsort(weights, axis=1)[::-1]
            top_names = np.take(in_names, order[:, :max_features])
            ix0 = np.arange(weights.shape[0]).repeat(max_features)
            ix0 = ix0.reshape(-1, max_features)
            top_weights = orig_weights[ix0, order[:, :max_features]]
            all_feats = (feat_join.join(feat_fmt.format(name=n, weight=w)
                                        for n, w in zip(names, weights)
                                        if w > threshold)
                         for names, weights
                         in zip(top_names.tolist(), top_weights.tolist()))
        else:
            all_feats = itertools.repeat('', weights.shape[0])

        return [fmt.format(func_name=func_name, i=i, feats=feats)
                for i, feats in enumerate(all_feats)]

    return transform_names


def make_tfn_tree_paths(get_trees, show_thresholds=True,
                        feature_order='depth'):
    """Constructs a feature name transformer for each leaf of each tree

    Parameters
    ----------
    get_trees : callable or str
        A function or attribute name which, applied to the registered
        transformer, will return one or more trees.
    show_thresholds : bool or str
        If False, feature names in the path are shown without split thresholds.
        If True (default), format string is '0.2g', otherwise according to
        provided string.
    feature_order : {'depth' (default), 'index', 'alpha'}
        'depth' uses the tree depth order, 'index' sorts by input feature name
        order, 'alpha' sorts alphabetically.
    """
    get_trees = _attrgetter_or_identity(get_trees)
    if show_thresholds is True:
        show_thresholds = '0.2g'
    threshold_fmt = '{:%s}' % show_thresholds

    def transform_names(est, in_names=None):
        trees = get_trees(est)
        try:
            iter(trees)
        except TypeError:
            trees = [trees]

        out = []
        for i, tree in enumerate(trees):
            if i == 0:
                in_names = _get_feature_names(est, feature_names=in_names,
                                              num_features=tree.n_features)

            # we use node_count + 1 to handle -1 indices
            parent = np.empty(tree.node_count + 1, dtype=int)
            # calculate parents
            parent[tree.children_left] = np.arange(tree.node_count)
            parent[tree.children_right] = np.arange(tree.node_count)
            parent[0] = -1

            # Note parent_feature[0] is junk
            parent_feature = tree.feature.take(parent[:-1], mode='clip')
            parent_feature_name = [in_names[f] for f in parent_feature]
            if show_thresholds:
                threshold_str = [threshold_fmt.format(t)
                                 for t in tree.threshold]
                parent_threshold_str = [threshold_str[p] for p in parent[:-1]]
                side = np.ones(tree.node_count + 1, dtype=int)
                side[tree.children_left] = -1
                parent_feature_name = [name + ('<=' if s == -1 else '>') + t
                                       for name, s, t
                                       in zip(parent_feature_name,
                                              side.tolist(),
                                              parent_threshold_str)]

            for i in np.flatnonzero(tree.children_left == -1):
                leaf_names = []
                leaf_features = []
                while i != 0:
                    leaf_names.append(parent_feature_name[i])
                    leaf_features.append(parent_feature[i])
                    i = parent[i]

                if feature_order == 'depth':
                    leaf_names = leaf_names[::-1]
                elif feature_order == 'alpha':
                    leaf_names.sort()
                elif feature_order == 'index':
                    leaf_names = np.take(leaf_names,
                                         np.argsort(leaf_features)).tolist()
                out.append('&'.join(leaf_names))

        return out

    return transform_names


def transform_feature_names_polynomial_features(est, in_names=None):
    # Thanks to @amueller in scikit-learn#6732
    powers = est.powers_
    if in_names is None:
        in_names = _get_feature_names(est, feature_names=in_names,
                                      num_features=powers.shape[1])
    feature_names = []
    for row in powers:
        inds = np.where(row)[0]
        if len(inds):
            name = "*".join("%s^%d" % (in_names[ind], exp)
                            if exp != 1 else in_names[ind]
                            for ind, exp in zip(inds, row[inds]))
        else:
            name = "1"
        feature_names.append(name)
    return feature_names


def transform_feature_names_binarizer(est, in_names=None):
    # XXX: does not handle in_names=None case
    fmt = '{{}}>{:.2g}'.format(est.threshold)
    return [fmt.format(name) for name in in_names]


# By default these are ignored as all features are treated identically
# An alternative is to use make_tfn_featurewise

@transform_feature_names.register(Imputer)
@transform_feature_names.register(StandardScaler)
@transform_feature_names.register(RobustScaler)
@transform_feature_names.register(MinMaxScaler)
@transform_feature_names.register(Normalizer)
def identity(est, in_names=None):
    return in_names


# Other preprocessing:

transform_feature_names.register(Binarizer)(
    transform_feature_names_binarizer)  # Perhaps should also be identity
transform_feature_names.register(PolynomialFeatures)(
    transform_feature_names_polynomial_features)
transform_feature_names.register(RandomTreesEmbedding)(
    make_tfn_tree_paths(lambda est: [dtc.tree_
                                     for dtc in est.estimators_]))

# TODO: OneHotEncoder


# Decomposition (linear weights):

transform_feature_names.register(TfidfTransformer)(
    make_tfn_featurewise('tfidf', lambda est: len(est.idf_)))
for cls, prefix in [(LatentDirichletAllocation, 'topic'),
                    (PCA, 'pc'), (TruncatedSVD, 'pc')]:
    transform_feature_names.register(cls)(
        make_tfn_weighted('components_', func_name=prefix))


# Feature selection:

@transform_feature_names.register(SelectorMixin)
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
