"""transform_feature_names implementations for scikit-learn transformers
"""

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection.base import SelectorMixin
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import Imputer

from eli5.transform import transform_feature_names


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


@transform_feature_names.register(SelectorMixin)
def _select_names(est, in_names=None):
    return [in_names[i] for i in est.get_support(indices=True)]


def _formatted_names(fmt):
    def transform_names(self, in_names=None):
        return [fmt.format(name) for name in in_names]
    return transform_names


def _component_names(fmt, attr):
    def transform_names(self, in_names=None):
        return [fmt.format(i) for i in range(getattr(self, attr))]
    return transform_names


transform_feature_names.register(TfidfTransformer)(
    _formatted_names('tfidf({})'))
transform_feature_names.register(Imputer)(
    _formatted_names('impute({})'))
transform_feature_names.register(LatentDirichletAllocation)(
    _component_names('topic({})', 'n_topics'))
