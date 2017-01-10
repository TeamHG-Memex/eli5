.. _library-sklearn-crfsuite:

sklearn-crfsuite
================

sklearn-crfsuite_ is a sequence classification library. It provides
a higher-level API for python-crfsuite_; python-crfsuite_ is a Python binding
for CRFSuite_ C++ library.

.. _sklearn-crfsuite: https://github.com/TeamHG-Memex/sklearn-crfsuite
.. _python-crfsuite: https://github.com/scrapinghub/python-crfsuite
.. _CRFSuite: https://github.com/chokkan/crfsuite

eli5 supports :func:`eli5.explain_weights` for `sklearn_crfsuite.CRF`_ objects;
explanation contains transition features table and state features table.

::

    import eli5
    eli5.explain_weights(crf)

See the :ref:`tutorial <sklearn-crfsuite-tutorial>` for a more detailed usage
example.

.. note::
    Top-level :func:`eli5.explain_weights` calls are dispatched
    to :func:`eli5.sklearn_crfsuite.explain_weights.explain_weights_sklearn_crfsuite`.

.. _sklearn_crfsuite.CRF: http://sklearn-crfsuite.readthedocs.io/en/latest/api.html#sklearn_crfsuite.CRF
