# -*- coding: utf-8 -*-
from __future__ import absolute_import
from singledispatch import singledispatch

import numpy as np
from sklearn.linear_model import (
    LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier,
    Perceptron, SGDClassifier,  SGDRegressor)
from sklearn.svm import LinearSVC
# TODO: see https://github.com/scikit-learn/scikit-learn/pull/2250
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import (
    GradientBoostingClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier

from eli5._feature_weights import get_top_features_dict
from eli5.utils import argsort_k_largest
from eli5.sklearn.utils import (
    get_coef,
    is_multiclass_classifier,
    is_multitarget_regressor,
    get_feature_names,
    rename_label,
)


LINEAR_CAVEATS = """
Caveats:
1. Be careful with features which are not
   independent - weights don't show their importance.
2. If scale of input features is different then scale of coefficients
   will also be different, making direct comparison between coefficient values
   incorrect.
3. Depending on regularization, rare features sometimes may have high
   coefficients; this doesn't mean they contribute much to the
   classification result for most examples.
""".lstrip()

DESCRIPTION_CLF_MULTICLASS = """
Features with largest coefficients per class.
""" + LINEAR_CAVEATS

DESCRIPTION_CLF_BINARY = """
Features with largest coefficients.
""" + LINEAR_CAVEATS

DESCRIPTION_REGRESSION = DESCRIPTION_CLF_BINARY

DESCRIPTION_REGRESSION_MULTITARGET = """
Features with largest coefficients per target.
""" + LINEAR_CAVEATS

DESCRIPTION_RANDOM_FOREST = """
Random forest feature importances; values are numbers 0 <= x <= 1;
all values sum to 1.
"""

DESCRIPTION_DECISION_TREE = """
Decision tree feature importances; values are numbers 0 <= x <= 1;
all values sum to 1.
"""

_TOP = 20


@singledispatch
def explain_weights(clf, vec=None, top=_TOP, class_names=None,
                    feature_names=None):
    """ Return an explanation of a classifier """
    return {
        "classifier": repr(clf),
        "description": "Error: classifier is not supported",
    }


@explain_weights.register(LogisticRegression)
@explain_weights.register(LogisticRegressionCV)
@explain_weights.register(SGDClassifier)
@explain_weights.register(PassiveAggressiveClassifier)
@explain_weights.register(Perceptron)
@explain_weights.register(LinearSVC)
def explain_linear_classifier_weights(clf, vec=None, top=_TOP, class_names=None,
                                      feature_names=None):
    """
    Return an explanation of a linear classifier weights in the following
    format::

        {
            "classifier": "<classifier repr>",
            "method": "<interpretation method>",
            "description": "<human readable description>",
            "classes": [
                {
                    "class": <class name>,
                    "feature_weights": [
                        {
                            # positive weights
                            "pos": [
                                (feature_name, coefficient),
                                ...
                            ],

                            # negative weights
                            "neg": [
                                (feature_name, coefficient),
                                ...
                            ],

                            # A number of features not shown
                            "pos_remaining": <int>,
                            "neg_remaining": <int>,

                            # Sum of feature weights not shown
                            # "pos_remaining_sum": <float>,
                            # "neg_remaining_sum": <float>,
                        },
                        ...
                    ]
                },
                ...
            ]
        }

    To print it use utilities from eli5.formatters.
    """
    feature_names = get_feature_names(clf, vec, feature_names=feature_names)

    def _features(label_id):
        coef = get_coef(clf, label_id)
        return get_top_features_dict(feature_names, coef, top)

    def _label(label_id, label):
        return rename_label(label_id, label, class_names)

    if is_multiclass_classifier(clf):
        return {
            'classes': [
                {
                    'class': _label(label_id, label),
                    'feature_weights': _features(label_id)
                }
                for label_id, label in enumerate(clf.classes_)
            ],
            'description': DESCRIPTION_CLF_MULTICLASS,
            'classifier': repr(clf),
            'method': 'linear model',
        }
    else:
        # for binary classifiers scikit-learn stores a single coefficient
        # vector, which corresponds to clf.classes_[1].
        return {
            'classes': [{
                'class': _label(1, clf.classes_[1]),
                'feature_weights': _features(0),
            }],
            'description': DESCRIPTION_CLF_BINARY,
            'classifier': repr(clf),
            'method': 'linear model',
        }


@explain_weights.register(RandomForestClassifier)
@explain_weights.register(ExtraTreesClassifier)
@explain_weights.register(GradientBoostingClassifier)
@explain_weights.register(AdaBoostClassifier)
def explain_rf_feature_importance(clf, vec, top=_TOP, class_names=None,
                                  feature_names=None):
    """
    Return an explanation of a tree-based ensemble classifier in the
    following format::

        {
            "classifier": "<classifier repr>",
            "method": "<interpretation method>",
            "description": "<human readable description>",
            "feature_importances": [
                (feature_name, importance, std_deviation),
                ...
            ]
        }
    """
    feature_names = get_feature_names(clf, vec, feature_names=feature_names)
    coef = clf.feature_importances_
    trees = np.array(clf.estimators_).ravel()
    coef_std = np.std([tree.feature_importances_ for tree in trees], axis=0)

    indices = argsort_k_largest(coef, top)
    names, values, std = feature_names[indices], coef[indices], coef_std[indices]
    return {
        'feature_importances': list(zip(names, values, std)),
        'description': DESCRIPTION_RANDOM_FOREST,
        'classifier': repr(clf),
        'method': 'feature importances',
    }


@explain_weights.register(DecisionTreeClassifier)
def explain_tree_feature_importance(clf, vec=None, top=_TOP, class_names=None,
                                    feature_names=None):
    """
    TODO/FIXME: should it be a tree instead?

    Return an explanation of a decision tree classifier in the
    following format (compatible with random forest explanations)::

        {
            "classifier": "<classifier repr>",
            "method": "<interpretation method>",
            "description": "<human readable description>",
            "feature_importances": [
                (feature_name, importance, std_deviation),
                ...
            ]
        }

    """
    feature_names = get_feature_names(clf, vec, feature_names=feature_names)
    coef = clf.feature_importances_
    indices = argsort_k_largest(coef, top)
    names, values = feature_names[indices], coef[indices]
    std = np.zeros_like(values)
    return {
        'feature_importances': list(zip(names, values, std)),
        'description': DESCRIPTION_DECISION_TREE,
        'classifier': repr(clf),
        'method': 'feature importances',
    }


@explain_weights.register(SGDRegressor)
def explain_linear_regressor_weights(clf, vec=None, feature_names=None, top=_TOP,
                                     target_names=None):
    """
    Return an explanation of a linear regressor weights in the following
    format::

        {
            "classifier": "<classifier repr>",
            "method": "<interpretation method>",
            "description": "<human readable description>",
            "targets": [
                {
                    "targets <target name>,
                    "feature_weights": [
                        {
                            # positive weights
                            "pos": [
                                (feature_name, coefficient),
                                ...
                            ],

                            # negative weights
                            "neg": [
                                (feature_name, coefficient),
                                ...
                            ],

                            # A number of features not shown
                            "pos_remaining": <int>,
                            "neg_remaining": <int>,

                            # Sum of feature weights not shown
                            # "pos_remaining_sum": <float>,
                            # "neg_remaining_sum": <float>,
                        },
                        ...
                    ]
                },
                ...
            ]
        }

    To print it use utilities from eli5.formatters.
    """
    feature_names = get_feature_names(clf, vec, feature_names=feature_names)

    def _features(target_id):
        coef = get_coef(clf, target_id)
        return get_top_features_dict(feature_names, coef, top)

    def _label(target_id, target):
        return rename_label(target_id, target, target_names)

    if is_multitarget_regressor(clf):
        return {
            'targets': [
                {
                    'target': _label(target_id, target),
                    'feature_weights': _features(target_id)
                }
                # FIXME - what if target_names is not set
                for target_id, target in enumerate(target_names)
                ],
            'description': DESCRIPTION_REGRESSION_MULTITARGET,
            'classifier': repr(clf),
            'method': 'linear model',
        }
    else:
        return {
            'targets': [{
                'target': _label(0, 'y'),
                'feature_weights': _features(0),
            }],
            'description': DESCRIPTION_REGRESSION,
            'classifier': repr(clf),
            'method': 'linear model',
        }
