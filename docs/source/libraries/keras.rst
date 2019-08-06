.. _library-keras:

Keras
=====

Keras_ is "a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano". 

Keras can be used for many Machine Learning tasks, and it has support for both popular
and experimental neural network architectures.

.. _Keras: https://keras.io/

.. _GradCAM: https://arxiv.org/abs/1610.02391/


.. _keras-explain-prediction:

explain_prediction
------------------

Currently ELI5 supports :func:`eli5.explain_prediction` for Keras image and text classifiers.
Explanations are done using the GradCAM_ technique. Using it, we differentiate the model's output with respect to a hidden layer. This results in a tensor of gradients. We take a linear combination of those gradients with the hidden layer's activations (output), getting a Grad-CAM "heatmap" (or "localization map"). The heatmap highlights what parts of the input (the hidden layer but resized) contributed to the prediction.


# TODO: show Grad-CAM "formula"


Important arguments to :func:`eli5.explain_prediction` when using with ``Model`` and ``Sequential``:

* ``model`` is the neural network model to be explained.

* ``doc`` input tensor into the network that will be explained.
    
    - **The tensor must be an instance of ``numpy.ndarray``.**

    - Usually the tensor has the format `(batch, height, width, channels)` for images and `(batch, series)` for text.
    
    - Check ``model.input_shape`` to see the required dimensions of the input tensor for your model.

    - **Currently only the first sample in a batch is explained. It's best to pass input with `batch=1`.**

    - **Currently only "channels last" tensor format is supported.**

* ``targets`` are the output classes to focus on. Possible values include: 

    - A list of integers (class ID's). **Only the first prediction from the list is currently taken. The list must be length one.**

    - `None` for automatically taking the top prediction that the model makes.

* ``layer`` is the layer in the model from which the heatmap will be generated. Possible values are:
    
    - An instance of ``Layer``, a layer name (str), or a layer index (int).

    - `None` for automatically getting a suitable layer, if possible.

* ``relu`` whether to apply `ReLU <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ to the heatmap.
    
    - The GradCAM_ paper applies ReLU to the produced heatmap in order to only show what increases a prediction.

    - Set to `False` in order to see negative values in the heatmap. This lets you see what makes the class score go down, or other classes present in the input.

* ``counterfactual`` whether to negate gradients when computing the heatmap.

    - The GradCAM_ paper mentions "counterfactual explanations". Such explanations show what makes the predicted class score go down. For example, this highlights other classes that are present in the input.

    - Set to `True` to produce counterfactual explanations.


Extra arguments for image-based explanations:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``image`` Pillow image, corresponds to ``doc`` input.

    - **Must be passed for image explanations.**

    - **Must have mode 'RGBA'.**

Image explanations are dispatched to :func:`eli5.keras.explain_prediction.explain_prediction_keras_image`.


Extra arguments for text-based explanations:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``tokens`` array of strings, corresponding to ``doc`` input.

    - **Must be passed for text explanations.** This is what will be highlighted for text explanations. Each token should correspond to an integer in ``doc``.

    - List or numpy array containing strings. For example, ``['a', 'sample', 'input']`` 

    - May have a batch dimension (i.e. numpy array with shape (numsamples, len), or a list of lists). *Note that only the first sample in the batch is currently explained.*

    - **Must be the same length as** ``doc``.

    - **If passing without batch dimension,** ``doc`` **must have batch size 1.**

    - May have padding if ``doc`` has padding.

* ``pad_value`` padding symbol.

    - Pass ``pad_value`` and ``padding`` in order to remove padding from the explanation.

    - Number inside ``doc`` or string inside ``tokens`` that is used to indicate padding.

    - For example, ``'<PAD>'`` or ``0``.

* ``padding`` padding location.

    - Either ``post`` for padding after the actual text starts, or ``pre`` for padding before the text starts.

* ``interpolation_kind`` method for resizing the heatmap to fit over input.

    - ``scipy`` interpolation method as a string.

    - See ``kind`` argument to `interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_.

    - Default is ``linear``.

Text explanations are dispatched to :func:`eli5.keras.explain_prediction.explain_prediction_keras_text`.


All other arguments are ignored.


:func:`eli5.explain_prediction` return value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An :class:`eli5.base.Explanation` instance is returned with some important attributes:

* ``image`` if explaining image-based networks, represents the image input into the model. A Pillow image with mode "RGBA".

* ``targets`` represents the explanation values for each target class (currently only 1 target is supported). A list of :class:`eli5.base.TargetExplanation` objects with the following attributes set:

    * ``heatmap``  is a "localization map" (a numpy array with float values). The numbers indicate how important the region in the image is for the target class (even if the target class was not the predicted class). Higher numbers mean that the region tends to increase the predicted value for a class. Lower numbers mean that the region has smaller effect on the predicted class score.
        
        - is a 2D numpy array for images.

        - is a 1D numpy array for text.

    * ``target`` the integer ID of the class explained (same as the argument to ``targets`` if one was passed, or the predicted class ID if no argument was passed).

    * ``score`` the output of the network for the predicted class.

    * ``weighted_spans`` an :class:`eli5.base.WeightedSpans` instance, if explaining text-based networks, text to be highlighted and the corresponding weights.


If neither ``image`` nor ``tokens`` are passed, an error explanation is returned.


.. note::
    Top-level :func:`eli5.explain_prediction` calls are dispatched
    to :func:`eli5.keras.explain_prediction.explain_prediction_keras` for
    ``keras.models.Model`` and ``keras.models.Sequential``.


.. _keras-show-prediction:

show_prediction
---------------

ELI5 supports :func:`eli5.show_prediction` to conveniently display explanations in an IPython cell.
:func:`eli5.explain_prediction` is called on a Keras model and the result is passed to a formatter.

For images, formatting is dispatched to :func:`eli5.format_as_image`.
For text, formatting is dispatched to :func:`eli5.format_as_html`.


.. _keras-gradcam:

Grad-CAM
--------

ELI5 contains :func:`eli5.keras.gradcam.gradcam_backend_keras`.

This function can be used to obtain the gradients and activations that are later used when computing a Grad-CAM heatmap.