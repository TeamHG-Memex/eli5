.. _library-keras:

Keras
=====

Keras_ is "a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano". 

Currently ELI5 supports :func:`eli5.explain_prediction` for Keras image classifiers.

.. _Keras: https://keras.io/

:func:`eli5.explain_prediction` explains image classifications through `Grad-CAM <https://arxiv.org/pdf/1610.02391.pdf>`_. The :class:`eli5.base.Explanation` object returned has a ``.image`` attribute that represents the  image that is inputted into the model, and a ``.heatmap`` attribute that is a grayscale "localization map" roughly indicating regions of importance in the image for the predicted class.

Important arguments for ``Model`` and ``Sequential``:

* ``doc`` is an image as a tensor that can be inputted to the model.

* ``target_names`` are the names of the output classes. 
    
    - *Currently not implemented*.

* ``targets`` are the output classes to focus on. Possible values include: 

    - A list of integers (class ID's). *Only the first prediction from the list is currently taken*. The list must be length one. 

    - None for automatically taking the top prediction of the model.

* ``layer`` is the layer in the model from which the heatmap will be generated. Possible values are:
    
    - An instance of ``Layer``, a name (str), or an index (int)

    - None for automatically getting a suitable layer if possible.

All other arguments are ignored.

.. note::
    Top-level :func:`eli5.explain_prediction` calls are dispatched
    to :func:`eli5.keras.explain_prediction_keras` for
    ``keras.models.Model`` and ``keras.models.Sequential``.

