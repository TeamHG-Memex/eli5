# -*- coding: utf-8 -*-
"""
Dispatch module. Explanation functions for conctere estimator classes
are defined in submodules.
"""
from singledispatch import singledispatch

from eli5.base import Explanation


@singledispatch
def explain_weights(estimator, **kwargs):
    """ Return an explanation of estimator parameters (weights).

    :func:`explain_weights` is not doing any work itself, it dispatches
    to a concrete implementation based on estimator type.

    Parameters
    ----------
    estimator : object
        Estimator instance. This argument must be positional.

    top : int or (int, int) tuple, optional
        Number of features to show. When ``top`` is int, ``top`` features with
        a highest absolute values are shown. When it is (pos, neg) tuple,
        no more than ``pos`` positive features and no more than ``neg``
        negative features is shown.

        This argument may be supported or not, depending on estimator type.

    target_names : list[str] or {'old_name': 'new_name'} dict, optional
        Names of targets or classes. This argument can be used to provide
        human-readable class/target names for estimators which don't expose
        clss names themselves. It can be also used to rename estimator-provided
        classes before displaying them.

        This argument may be supported or not, depending on estimator type.

    target_order : list, optional
        Order of class/target names to show. This argument can be also used
        to show information only for a subset of classes. It should be a list
        of class / target names which match names provided by an estimator
        (i.e. not names defined by ``target_names`` parameter).

        This argument may be supported or not, depending on estimator type.

    feature_names : list, optional
        A list of feature names. It allows to specify feature
        names when they are not provided by an estimator object.

        This argument may be supported or not, depending on estimator type.

    **kwargs: dict
        Keyword arguments. All keyword arguments are passed to
        concrete explain_weights... implementations.

    Returns
    -------
    Explanation
        :class:`~Explanation` result. Use one of the formatting functions from
        :mod:`eli5.formatters` to print it in a human-readable form.
        Explanation instances also have repr which works well with
        IPython notebook.
    """
    return Explanation(
        estimator=repr(estimator),
        error="estimator %r is not supported" % estimator,
    )


@singledispatch
def explain_prediction(estimator, doc, **kwargs):
    """
    Return an explanation of an estimator prediction.

    :func:`explain_prediction` is not doing any work itself, it dispatches
    to a concrete implementation based on estimator type.

    Parameters
    ----------
    estimator : object
        Estimator instance. This argument must be positional.

    doc : object
        Example to run estimator on. Estimator makes a prediction for this
        example, and :func:`explain_prediction` tries to show information
        about this prediction.

    top : int or (int, int) tuple, optional
        Number of features to show. When ``top`` is int, ``top`` features with
        a highest absolute values are shown. When it is (pos, neg) tuple,
        no more than ``pos`` positive features and no more than ``neg``
        negative features is shown.

        This argument may be supported or not, depending on estimator type.

    target_names : list[str] or {'old_name': 'new_name'} dict, optional
        Names of targets or classes. This argument can be used to provide
        human-readable class/target names for estimators which don't expose
        clss names themselves. It can be also used to rename estimator-provided
        classes before displaying them.

        This argument may be supported or not, depending on estimator type.

    target_order : list, optional
        Order of class/target names to show. This argument can be also used
        to show information only for a subset of classes. It should be a list
        of class / target names which match names provided by an estimator
        (i.e. not names defined by ``target_names`` parameter).

        This argument may be supported or not, depending on estimator type.

    feature_names : list, optional
        A list of feature names. It allows to specify feature
        names when they are not provided by an estimator object.

        This argument may be supported or not, depending on estimator type.

    **kwargs: dict
        Keyword arguments. All keyword arguments are passed to
        concrete explain_prediction... implementations.

    Returns
    -------
    Explanation
        :class:`~.Explanation` result. Use one of the formatting functions from
        :mod:`eli5.formatters` to print it in a human-readable form.
        Explanation instances also have repr which works well with
        IPython notebook.
    """
    return Explanation(
        estimator=repr(estimator),
        error="estimator %r is not supported" % estimator,
    )
