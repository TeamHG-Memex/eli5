.. module:: eli5.formatters

eli5.formatters
===============

This module holds functions that convert :class:`~.Explanation` objects
(returned by :func:`eli5.explain_weights` and :func:`eli5.explain_prediction`)
into HTML, text, dict/JSON or pandas DataFrames. The following functions are
also available in ``eli5`` namespace (e.g. ``eli5.formas_as_html``):

* :func:`eli5.formatters.html.format_as_html`
* :func:`eli5.formatters.html.format_html_styles`
* :func:`eli5.formatters.text.format_as_text`
* :func:`eli5.formatters.as_dict.format_as_dict`
* :func:`eli5.formatters.as_dataframe.explain_weights_df`
* :func:`eli5.formatters.as_dataframe.explain_weights_dfs`
* :func:`eli5.formatters.as_dataframe.explain_prediction_df`
* :func:`eli5.formatters.as_dataframe.explain_prediction_dfs`
* :func:`eli5.formatters.as_dataframe.format_as_dataframe`
* :func:`eli5.formatters.as_dataframe.format_as_dataframes`


eli5.formatters.html
--------------------

.. automodule:: eli5.formatters.html
    :members:

eli5.formatters.text
--------------------

.. automodule:: eli5.formatters.text
    :members:

eli5.formatters.as_dict
-----------------------

.. automodule:: eli5.formatters.as_dict
    :members:

eli5.formatters.as_dataframe
----------------------------

.. automodule:: eli5.formatters.as_dataframe
    :members:
