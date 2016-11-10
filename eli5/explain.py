# -*- coding: utf-8 -*-
"""
Dispatch module. Explanation functions for conctere estimator classes
are defined in submodules.
"""
from singledispatch import singledispatch

from eli5.base import Explanation


@singledispatch
def explain_prediction(estimator, doc, **kwargs):
    """ Return an explanation of an estimator prediction """
    return Explanation(
        estimator=repr(estimator),
        error="estimator %r is not supported" % estimator,
    )


@singledispatch
def explain_weights(estimator, **kwargs):
    """ Return an explanation of an estimator weights """
    return Explanation(
        estimator=repr(estimator),
        error="estimator %r is not supported" % estimator,
    )
