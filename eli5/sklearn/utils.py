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


def get_feature_names(clf, vec=None, bias_name='<BIAS>', feature_names=None):
    """
    Return a vector of feature names, including bias feature.
    If vec is None or doesn't have get_feature_names() method,
    features are named x1, x2, etc.
    """
    if feature_names is None:
        if vec and hasattr(vec, 'get_feature_names'):
            feature_names = vec.get_feature_names()
        else:
            num_features = clf.coef_[0].shape[0]
            feature_names = ["x%d" % i for i in range(num_features)]
        if bias_name is not None and has_intercept(clf):
            feature_names += [bias_name]
    else:
        num_features = clf.coef_[0].shape[0] + int(has_intercept(clf))
        if len(feature_names) != num_features:
            raise ValueError("feature_names has a wrong lenght: "
                             "expected=%d, got=%d" % (num_features,
                                                      len(feature_names)))
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
