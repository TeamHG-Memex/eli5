# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np  # type: ignore
from scipy import sparse as sp  # type: ignore
from sklearn_crfsuite import CRF  # type: ignore

from eli5.base import Explanation, TargetExplanation, TransitionFeatureWeights
from eli5.explain import explain_weights
from eli5.utils import get_target_display_names
from eli5._feature_names import FeatureNames
from eli5._feature_weights import get_top_features


@explain_weights.register(CRF)
def explain_weights_sklearn_crfsuite(crf,
                                     top=20,
                                     target_names=None,
                                     targets=None,
                                     feature_re=None,
                                     feature_filter=None):
    """ Explain sklearn_crfsuite.CRF weights.

    See :func:`eli5.explain_weights` for description of
    ``top``, ``target_names``, ``targets``,
    ``feature_re`` and ``feature_filter`` parameters.
    """
    feature_names = np.array(crf.attributes_)
    state_coef = crf_state_coef(crf).todense().A
    transition_coef = crf_transition_coef(crf)

    if feature_filter is not None or feature_re is not None:
        state_feature_names, flt_indices = (
            FeatureNames(feature_names).handle_filter(feature_filter, feature_re))
        state_feature_names = np.array(state_feature_names.feature_names)
        state_coef = state_coef[:, flt_indices]
    else:
        state_feature_names = feature_names

    def _features(label_id):
        return get_top_features(state_feature_names, state_coef[label_id], top)

    if targets is None:
        targets = sorted_for_ner(crf.classes_)

    display_names = get_target_display_names(crf.classes_, target_names,
                                             targets)
    indices, names = zip(*display_names)
    transition_coef = filter_transition_coefs(transition_coef, indices)

    return Explanation(
        targets=[
            TargetExplanation(
                target=label,
                feature_weights=_features(label_id)
            )
            for label_id, label in zip(indices, names)
        ],
        transition_features=TransitionFeatureWeights(
            class_names=names,
            coef=transition_coef,
        ),
        estimator=repr(crf),
        method='CRF',
    )


def crf_state_coef(crf):
    attr_index = {name: idx for idx, name in enumerate(crf.attributes_)}
    class_index = {cls_name: idx for idx, cls_name in enumerate(crf.classes_)}

    n_features = len(crf.attributes_)
    n_classes = len(crf.classes_)
    coef = sp.dok_matrix((n_classes, n_features))

    for (feat, cls), value in crf.state_features_.items():
        coef[class_index[cls], attr_index[feat]] = value

    return coef.tocsr()


def crf_transition_coef(crf):
    n_classes = len(crf.classes_)
    coef = np.empty((n_classes, n_classes))

    for i, cls_from in enumerate(crf.classes_):
        for j, cls_to in enumerate(crf.classes_):
            w = crf.transition_features_.get((cls_from, cls_to), 0)
            coef[i, j] = w

    return coef


def filter_transition_coefs(transition_coef, indices):
    """
    >>> coef = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> filter_transition_coefs(coef, [0])
    array([[0]])
    >>> filter_transition_coefs(coef, [1, 2])
    array([[4, 5],
           [7, 8]])
    >>> filter_transition_coefs(coef, [2, 0])
    array([[8, 6],
           [2, 0]])
    >>> filter_transition_coefs(coef, [0, 1, 2])
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    """
    indices = np.array(indices)
    rows = transition_coef[indices]
    return rows[:,indices]


def sorted_for_ner(crf_classes):
    """
    Return labels sorted in a default order suitable for NER tasks:

    >>> sorted_for_ner(['B-ORG', 'B-PER', 'O', 'I-PER'])
    ['O', 'B-ORG', 'B-PER', 'I-PER']
    """
    def key(cls):
        if len(cls) > 2 and cls[1] == '-':
            # group names like B-ORG and I-ORG together
            return cls.split('-', 1)[1], cls
        return '', cls
    return sorted(crf_classes, key=key)
