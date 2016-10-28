# -*- coding: utf-8 -*-
"""
Dispatch module. Explanation functions for conctere estimator classes
are defined in submodules.
"""
from singledispatch import singledispatch


@singledispatch
def explain_prediction(estimator, doc, **kwargs):
    """ Return an explanation of an estimator """
    return {
        "estimator": repr(estimator),
        "description": "Error: estimator %r is not supported" % estimator,
    }


@singledispatch
def explain_weights(estimator, **kwargs):
    """ Return an explanation of an estimator """
    return {
        "estimator": repr(estimator),
        "description": "Error: estimator %r is not supported" % estimator,
    }
