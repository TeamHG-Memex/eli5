# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
from scipy import sparse as sp
from sklearn_crfsuite import CRF
from pycrfsuite import Trainer

from eli5.base import Explanation, TargetExplanation, TransitionFeatureWeights
from eli5.explain import explain_weights
from eli5._feature_weights import get_top_features


@explain_weights.register(CRF)
def explain_weights_sklearn_crfsuite(crf, top=20):
    feature_names = np.array(crf.attributes_)
    state_coef = crf_state_coef(crf).todense().A
    transition_coef = crf_transition_coef(crf)

    def _features(label_id):
        return get_top_features(feature_names, state_coef[label_id], top)

    return Explanation(
        targets=[
            TargetExplanation(
                target=label,
                feature_weights=_features(label_id)
            )
            for label_id, label in enumerate(crf.classes_)
        ],
        transition_features=TransitionFeatureWeights(
            class_names=crf.classes_,
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
