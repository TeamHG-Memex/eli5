# -*- coding: utf-8 -*-
import numpy as np


def is_multiclass_classifier(clf):
    """
    Return True if a classifier is multiclass or False if it is binary.
    """
    return clf.coef_.shape[0] > 1


def is_probabilistic_classifier(clf):
    """ Return True if a classifier can return probabilities """
    return hasattr(clf, 'predict_proba')


def has_intercept(clf):
    """ Return True if classifier has intercept fit. """
    return getattr(clf, 'fit_intercept', False)


def get_feature_names(clf, vec, bias_name='<BIAS>'):
    """ Return a vector of feature names, including bias feature """
    feature_names = vec.get_feature_names()
    if has_intercept(clf):
        feature_names += [bias_name]
    return np.array(feature_names)


def get_coef(clf, label_id):
    """
    Return a vector of coefficients for a given label,
    including bias feature.
    """
    coef = clf.coef_[label_id]  # multiclass case, also works for binary
    if not has_intercept(clf):
        return coef
    bias = clf.intercept_[label_id]
    return np.hstack([coef, bias])


def rename_label(label_id, label, class_names):
    """ Rename label according to class_names """
    if class_names is None:
        return label
    if isinstance(class_names, dict):
        return class_names[label]
    return class_names[label_id]
