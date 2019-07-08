.. _library-keras:

Keras
=====

Keras_ is "a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano". 

Keras can be used for many Machine Learning tasks, and it has support for both popular
and experimental neural network architectures.

.. _Keras: https://keras.io/


.. _keras-explain-prediction:

explain_prediction
------------------

Currently ELI5 supports :func:`eli5.explain_prediction` for Keras image classifiers.
:func:`eli5.explain_prediction` explains image classifications through `Grad-CAM <https://arxiv.org/pdf/1610.02391.pdf>`_.

The returned :class:`eli5.base.Explanation` object contains some important objects:

* ``image`` represents the image that is inputted into the model.

* ``heatmap``  (property of each :class:`eli5.base.TargetExplanation` in ``targets``) is a grayscale "localization map", roughly indicating regions of importance in the image for the predicted class.


Important arguments to :func:`eli5.explain_prediction` for ``Model`` and ``Sequential``:

* ``doc`` is an image as a tensor that can be inputted to the model.
    
    - The type is a ``numpy.ndarray``.

    - Usually the tensor has the format `(batch, dims, ..., channels)` (channels last format, `dims=(height, width)`, `batch=1`, one image), i.e. `BHWC`.
    
    - Check ``model.input_shape`` to confirm the required dimensions of the input tensor.

* ``target_names`` are the names of the output classes. 
    
    - *Currently not implemented*.

* ``targets`` are the output classes to focus on. Possible values include: 

    - A list of integers (class ID's). *Only the first prediction from the list is currently taken*. The list must be length one. 

    - None for automatically taking the top prediction of the model.

* ``layer`` is the layer in the model from which the heatmap will be generated. Possible values are:
    
    - An instance of ``Layer``, a name (str), or an index (int)

    - None for automatically getting a suitable layer, if possible.

All other arguments are ignored.


.. note::
    Top-level :func:`eli5.explain_prediction` calls are dispatched
    to :func:`eli5.keras.explain_prediction_keras` for
    ``keras.models.Model`` and ``keras.models.Sequential``.


.. _keras-show-prediction:

show_prediction
---------------

ELI5 supports :func:`eli5.show_prediction` to conveniently 
invoke ``explain_prediction`` with ``format_as_image``, and display the explanation in an
IPython cell.


.. _keras-gradcam:

Grad-CAM
--------

ELI5 contains :func:`eli5.keras.gradcam.gradcam` and :func:`eli5.keras.gradcam.gradcam_backend`.

These functions can be used to obtain finer details of a Grad-CAM explanation.