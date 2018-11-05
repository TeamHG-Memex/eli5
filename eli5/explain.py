# -*- coding: utf-8 -*-
"""
Dispatch module. Explanation functions for conctere estimator classes
are defined in submodules.
"""
from eli5.base import Explanation
from eli5.base_utils import singledispatch


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
        negative features is shown. ``None`` value means no limit.

        This argument may be supported or not, depending on estimator type.

    target_names : list[str] or {'old_name': 'new_name'} dict, optional
        Names of targets or classes. This argument can be used to provide
        human-readable class/target names for estimators which don't expose
        clss names themselves. It can be also used to rename estimator-provided
        classes before displaying them.

        This argument may be supported or not, depending on estimator type.

    targets : list, optional
        Order of class/target names to show. This argument can be also used
        to show information only for a subset of classes. It should be a list
        of class / target names which match either names provided by
        an estimator or names defined in ``target_names`` parameter.

        This argument may be supported or not, depending on estimator type.

    feature_names : list, optional
        A list of feature names. It allows to specify feature
        names when they are not provided by an estimator object.

        This argument may be supported or not, depending on estimator type.

    feature_re : str, optional
        Only feature names which match ``feature_re`` regex are returned
        (more precisely, ``re.search(feature_re, x)`` is checked).

    feature_filter : Callable[[str], bool], optional
        Only feature names for which ``feature_filter`` function returns True
        are returned.

    **kwargs: dict
        Keyword arguments. All keyword arguments are passed to
        concrete explain_weights... implementations.

    Returns
    -------
    Explanation
        :class:`~Explanation` result. Use one of the formatting functions from
        :mod:`eli5.formatters` to print it in a human-readable form.

        Explanation instances have repr which works well with
        IPython notebook, but it can be a better idea to use
        :func:`eli5.show_weights` instead of :func:`eli5.explain_weights`
        if you work with IPython: :func:`eli5.show_weights` allows to customize
        formatting without a need to import :mod:`eli5.formatters` functions.
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
        about this prediction. Pass a single element, not a one-element array:
        if you fitted your estimator on ``X``, that would be ``X[i]`` for
        most containers, and ``X.iloc[i]`` for ``pandas.DataFrame``.

    top : int or (int, int) tuple, optional
        Number of features to show. When ``top`` is int, ``top`` features with
        a highest absolute values are shown. When it is (pos, neg) tuple,
        no more than ``pos`` positive features and no more than ``neg``
        negative features is shown. ``None`` value means no limit (default).

        This argument may be supported or not, depending on estimator type.

    top_targets : int, optional
        Number of targets to show. When ``top_targets`` is provided,
        only specified number of targets with highest scores are shown.
        Negative value means targets with lowest scores are shown.
        Must not be given with ``targets`` argument.
        ``None`` value means no limit: all targets are shown (default).

        This argument may be supported or not, depending on estimator type.

    target_names : list[str] or {'old_name': 'new_name'} dict, optional
        Names of targets or classes. This argument can be used to provide
        human-readable class/target names for estimators which don't expose
        clss names themselves. It can be also used to rename estimator-provided
        classes before displaying them.

        This argument may be supported or not, depending on estimator type.

    targets : list, optional
        Order of class/target names to show. This argument can be also used
        to show information only for a subset of classes. It should be a list
        of class / target names which match either names provided by
        an estimator or names defined in ``target_names`` parameter.
        Must not be given with ``top_targets`` argument.

        In case of binary classification you can use this argument to
        set the class which probability or score should be displayed, with
        an appropriate explanation. By default a result for predicted class
        is shown. For example, you can use ``targets=[True]`` to always show
        result for a positive class, even if the predicted label is False.

        This argument may be supported or not, depending on estimator type.

    feature_names : list, optional
        A list of feature names. It allows to specify feature
        names when they are not provided by an estimator object.

        This argument may be supported or not, depending on estimator type.

    feature_re : str, optional
        Only feature names which match ``feature_re`` regex are returned
        (more precisely, ``re.search(feature_re, x)`` is checked).

    feature_filter : Callable[[str, float], bool], optional
        Only feature names for which ``feature_filter`` function returns True
        are returned. It must accept feature name and feature value.
        Missing features always have a NaN value.

    **kwargs: dict
        Keyword arguments. All keyword arguments are passed to
        concrete explain_prediction... implementations.

    Returns
    -------
    Explanation
        :class:`~.Explanation` result. Use one of the formatting functions from
        :mod:`eli5.formatters` to print it in a human-readable form.

        Explanation instances have repr which works well with
        IPython notebook, but it can be a better idea to use
        :func:`eli5.show_prediction` instead of :func:`eli5.explain_prediction`
        if you work with IPython: :func:`eli5.show_prediction` allows to
        customize formatting without a need to import :mod:`eli5.formatters`
        functions.
    """
    return Explanation(
        estimator=repr(estimator),
        error="estimator %r is not supported" % estimator,
    )
