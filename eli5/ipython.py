# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import Any, Dict, Tuple

from IPython.display import HTML  # type: ignore

from .explain import explain_weights, explain_prediction
from .formatters import format_as_html, fields


FORMAT_KWARGS = {'include_styles', 'force_weights',
                 'show', 'preserve_density',
                 'highlight_spaces', 'horizontal_layout',
                 'show_feature_values'}


def show_weights(estimator, **kwargs):
    """ Return an explanation of estimator parameters (weights)
    as an IPython.display.HTML object. Use this function
    to show classifier weights in IPython.

    :func:`show_weights` accepts all
    :func:`eli5.explain_weights` arguments and all
    :func:`eli5.formatters.html.format_as_html`
    keyword arguments, so it is possible to get explanation and
    customize formatting in a single call.

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
        Only feature names which match ``feature_re`` regex are shown
        (more precisely, ``re.search(feature_re, x)`` is checked).

    feature_filter : Callable[[str], bool], optional
        Only feature names for which ``feature_filter`` function returns True
        are shown.

    show : List[str], optional
        List of sections to show. Allowed values:

        * 'targets' - per-target feature weights;
        * 'transition_features' - transition features of a CRF model;
        * 'feature_importances' - feature importances of a decision tree or
          an ensemble-based estimator;
        * 'decision_tree' - decision tree in a graphical form;
        * 'method' - a string with explanation method;
        * 'description' - description of explanation method and its caveats.

        ``eli5.formatters.fields`` provides constants that cover common cases:
        ``INFO`` (method and description), ``WEIGHTS`` (all the rest),
        and ``ALL`` (all).

    horizontal_layout : bool
        When True, feature weight tables are printed horizontally
        (left to right); when False, feature weight tables are printed
        vertically (top to down). Default is True.

    highlight_spaces : bool or None, optional
        Whether to highlight spaces in feature names. This is useful if
        you work with text and have ngram features which may include spaces
        at left or right. Default is None, meaning that the value used
        is set automatically based on vectorizer and feature values.

    include_styles : bool
        Most styles are inline, but some are included separately in <style> tag;
        you can omit them by passing ``include_styles=False``. Default is True.

    **kwargs: dict
        Keyword arguments. All keyword arguments are passed to
        concrete explain_weights... implementations.

    Returns
    -------
    IPython.display.HTML
        The result is printed in IPython notebook as an HTML widget.
        If you need to display several explanations as an output of a single
        cell, or if you want to display it from a function then use
        IPython.display.display::

            from IPython.display import display
            display(eli5.show_weights(clf1))
            display(eli5.show_weights(clf2))

    """
    format_kwargs, explain_kwargs = _split_kwargs(kwargs)
    expl = explain_weights(estimator, **explain_kwargs)
    html = format_as_html(expl, **format_kwargs)
    return HTML(html)


def show_prediction(estimator, doc, **kwargs):
    """ Return an explanation of estimator prediction
    as an IPython.display.HTML object. Use this function
    to show information about classifier prediction in IPython.

    :func:`show_prediction` accepts all
    :func:`eli5.explain_prediction` arguments and all
    :func:`eli5.formatters.html.format_as_html`
    keyword arguments, so it is possible to get explanation and
    customize formatting in a single call.

    Parameters
    ----------
    estimator : object
        Estimator instance. This argument must be positional.

    doc : object
        Example to run estimator on. Estimator makes a prediction for this
        example, and :func:`show_prediction` tries to show information
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
        Only feature names which match ``feature_re`` regex are shown
        (more precisely, ``re.search(feature_re, x)`` is checked).

    feature_filter : Callable[[str, float], bool], optional
        Only feature names for which ``feature_filter`` function returns True
        are shown. It must accept feature name and feature value.
        Missing features always have a NaN value.

    show : List[str], optional
        List of sections to show. Allowed values:

        * 'targets' - per-target feature weights;
        * 'transition_features' - transition features of a CRF model;
        * 'feature_importances' - feature importances of a decision tree or
          an ensemble-based estimator;
        * 'decision_tree' - decision tree in a graphical form;
        * 'method' - a string with explanation method;
        * 'description' - description of explanation method and its caveats.

        ``eli5.formatters.fields`` provides constants that cover common cases:
        ``INFO`` (method and description), ``WEIGHTS`` (all the rest),
        and ``ALL`` (all).

    horizontal_layout : bool
        When True, feature weight tables are printed horizontally
        (left to right); when False, feature weight tables are printed
        vertically (top to down). Default is True.

    highlight_spaces : bool or None, optional
        Whether to highlight spaces in feature names. This is useful if
        you work with text and have ngram features which may include spaces
        at left or right. Default is None, meaning that the value used
        is set automatically based on vectorizer and feature values.

    include_styles : bool
        Most styles are inline, but some are included separately in <style> tag;
        you can omit them by passing ``include_styles=False``. Default is True.

    force_weights : bool
        When True, a table with feature weights is displayed even if all
        features are already highlighted in text. Default is False.

    preserve_density: bool or None
        This argument currently only makes sense when used with text data
        and vectorizers from scikit-learn.

        If preserve_density is True, then color for longer fragments will be
        less intensive than for shorter fragments, so that "sum" of intensities
        will correspond to feature weight.

        If preserve_density is None, then it's value is chosen depending on
        analyzer kind: it is preserved for "char" and "char_wb" analyzers,
        and not preserved for "word" analyzers.

        Default is None.

    show_feature_values : bool
        When True, feature values are shown along with feature contributions.
        Default is False.

    **kwargs: dict
        Keyword arguments. All keyword arguments are passed to
        concrete explain_prediction... implementations.

    Returns
    -------
    IPython.display.HTML
        The result is printed in IPython notebook as an HTML widget.
        If you need to display several explanations as an output of a single
        cell, or if you want to display it from a function then use
        IPython.display.display::

            from IPython.display import display
            display(eli5.show_weights(clf1))
            display(eli5.show_weights(clf2))
    """
    format_kwargs, explain_kwargs = _split_kwargs(kwargs)
    expl = explain_prediction(estimator, doc, **explain_kwargs)
    html = format_as_html(expl, **format_kwargs)
    return HTML(html)


def _split_kwargs(kwargs):
    # type: (Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]
    format_kwargs = {k: v for k, v in kwargs.items() if k in FORMAT_KWARGS}
    format_kwargs.setdefault('show', fields.WEIGHTS)
    format_kwargs.setdefault('force_weights', False)
    explain_kwargs = {k: v for k, v in kwargs.items() if k not in FORMAT_KWARGS}
    return format_kwargs, explain_kwargs
