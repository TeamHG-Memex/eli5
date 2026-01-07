.. _eli5-black-box:

Inspecting Black-Box Estimators
===============================

:func:`eli5.explain_weights` and :func:`eli5.explain_prediction` support
a lot of estimators and pipelines directly, but it is not possible to support
everything explicitly. So eli5 provides a way to inspect ML pipelines as black
boxes: :ref:`eli5-permutation-importance` method allows to use
:func:`eli5.explain_weights` with black-box estimators, while :ref:`eli5-lime`
allows to use :func:`eli5.explain_prediction`.

.. toctree::
   :maxdepth: 1

   lime
   permutation_importance
