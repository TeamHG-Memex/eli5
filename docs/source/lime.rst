.. _eli5-lime:

LIME
====

Algorithm
---------

LIME_ (Ribeiro et. al. 2016) is an algorithm to explain predictions
of black-box estimators:

1. Generate a fake dataset from the example we're going to explain.

2. Use black-box estimator to get target values for each example in a generated
   dataset (e.g. class probabilities).

3. Train a new white-box estimator, using generated dataset
   and generated labels as training data. It means we're trying to create
   an estimator which works the same as a black-box estimator, but which is
   easier to inspect. It doesn't have to work well globally, but it must
   approximate the black-box model well in the area close to the original
   example.

   To express "area close to the original example" user must provide
   a distance/similarity metric for examples in a generated dataset.
   Then training data is weighted according to a distance from the
   original example - the further is example, the less it affects weights
   of a white-box estimator.

4. Explain the original example through weights of this white-box estimator
   instead.

5. Prediction quality of a white-box classifer shows how well it approximates
   the black-box classifier. If the quality is low then explanation
   shouldn't be trusted.

.. _LIME: http://arxiv.org/abs/1602.04938

eli5.lime
---------

To understand how to use ``eli5.lime`` with text data check the
:ref:`TextExplainer tutorial <lime-tutorial>`. API reference is available
:mod:`here <eli5.lime>`. Currently eli5 doesn't provide a lot of helpers
for LIME + non-text data, but there is an IPyhton
`notebook <https://github.com/TeamHG-Memex/eli5/blob/master/notebooks/LIME%20and%20synthetic%20data.ipynb>`__
with an example of applying LIME for such tasks.

Caveats
-------

It sounds too good to be true, and indeed there are caveats:

1. If a white-box estimator gets a high score on a generated dataset
   it doesn't necessarily mean it could be trusted - it could also mean that
   the generated dataset is too easy and uniform, or that similarity
   metric provided by user assigns very low values for most examples,
   so that "area close to the original example" is too small to be interesting.

2. Fake dataset generation is the main issue; it is task-specific
   to a large extent. So LIME_ can work with any black-box classifier,
   but user may need to write code specific for each dataset.
   There is an opposite tradeoff in inspecting model weights:
   it works for any task, but one must write inspection code for each
   estimator type.

   eli5.lime provides dataset generation utilities for text data
   (remove random words) and for arbitrary data
   (sampling using Kernel Density Estimation).

   For text data eli5 also provides :class:`eli5.lime.TextExplainer`
   which brings together all LIME steps and allows to explain text classifiers;
   it still needs to make assumptions about the classifier in order to
   generate efficient fake dataset.

3. Similarity metric has a huge effect on a result. By choosing
   neighbourhood of a different size one can get opposite explanations.

Alternative implementations
---------------------------

There is a LIME implementation by LIME authors:
https://github.com/marcotcr/lime, so it is eli5.lime which should be considered
as alternative. At the time of writing eli5.lime has some differences from the
canonical LIME implementation:

1. eli5 supports many white-box classifiers from several libraries,
   you can use any of them with LIME;
2. eli5 supports dataset generation using Kernel Density Estimation,
   to ensure that generated dataset looks similar to the original dataset;
3. for explaining predictions of probabilistic classifiers
   eli5 uses another classifier by default, trained using cross-entropy loss,
   while canonical library fits regression model on probability output.

There are also features which are supported by original implementation,
but not by eli5, and the UIs are different.
