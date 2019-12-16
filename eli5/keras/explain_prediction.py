# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import Union, Optional, Callable, Tuple, List, TYPE_CHECKING
if TYPE_CHECKING:
    import PIL

import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Layer
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    GlobalMaxPooling2D,
    GlobalAveragePooling2D,
)
from keras.preprocessing.image import array_to_img

from eli5.base import Explanation, TargetExplanation
from eli5.explain import explain_prediction
from .gradcam import gradcam, gradcam_backend


DESCRIPTION_KERAS = """Grad-CAM visualization for image classification; 
output is explanation object that contains input image 
and heatmap image for a target.
"""

# note that keras.models.Sequential subclasses keras.models.Model
@explain_prediction.register(Model)
def explain_prediction_keras(model, # type: Model
                             doc, # type: np.ndarray
                             targets=None, # type: Optional[list]
                             layer=None, # type: Optional[Union[int, str, Layer]]
                             image=None,
                             ):
    # type: (...) -> Explanation
    """
    Explain the prediction of a Keras classifier with the Grad-CAM technique.

    We explicitly assume that the model's task is classification, i.e. final output is class scores.

    :param keras.models.Model model:
        Instance of a Keras neural network model,
        whose predictions are to be explained.

    :param numpy.ndarray doc:
        An input to ``model`` whose prediction will be explained.

        Currently only numpy arrays are supported.

        The tensor must be of suitable shape for the ``model``.

        Check ``model.input_shape`` to confirm the required dimensions of the input tensor.


        :raises TypeError: if ``doc`` is not a numpy array.
        :raises ValueError: if ``doc`` shape does not match.

    :param targets:
        Prediction ID's to focus on.

        *Currently only the first prediction from the list is explained*.
        The list must be length one.

        If None, the model is fed the input image and its top prediction
        is taken as the target automatically.


        :raises ValueError: if ``targets`` is a list with more than one item.
        :raises TypeError: if ``targets`` is not list or None.
    :type targets: list[int], optional

    :param layer:
        The activation layer in the model to perform Grad-CAM on:
        a valid keras layer name, layer index, or an instance of a Keras layer.

        If None, a suitable layer is attempted to be retrieved.
        For best results, pick a layer that:

        * has spatial or temporal information (conv, recurrent, pooling, embedding)
          (not dense layers).
        * shows high level features.
        * has large enough dimensions for resizing over input to work.


        :raises TypeError: if ``layer`` is not None, str, int, or keras.layers.Layer instance.
        :raises ValueError: if suitable layer can not be found.
        :raises ValueError: if differentiation fails with respect to retrieved ``layer``.
    :type layer: int or str or keras.layers.Layer, optional


    See :func:`eli5.explain_prediction` for more information about the ``model``,
    ``doc``, and ``targets`` parameters.


    Other arguments are passed to concrete implementations
    for image and text explanations.


    Returns
    -------
      expl : :class:`eli5.base.Explanation`
        An :class:`eli5.base.Explanation` object for the relevant implementation.
    """
    # Note that this function should only do dispatch
    # and no other processing
    if image is not None or _maybe_image(model, doc):
        return explain_prediction_keras_image(model,
                                              doc,
                                              image=image,
                                              targets=targets,
                                              layer=layer,
                                              )
    else:
        return explain_prediction_keras_not_supported(model, doc)


def explain_prediction_keras_not_supported(model, doc):
    """
    Can not do an explanation based on the passed arguments.
    Did you pass either "image" or "tokens"?
    """
    return Explanation(
        model.name,
        error='model "{}" is not supported, '
              'try passing the "image" argument if explaining an image model.'.format(model.name),
    )
    # TODO (open issue): implement 'other'/differentiable network type explanations


def explain_prediction_keras_image(model,
                                   doc,
                                   image=None, # type: Optional['PIL.Image.Image']
                                   targets=None,
                                   layer=None,
                                   ):
    """
    Explain an image-based model, highlighting what contributed in the image.

    :param numpy.ndarray doc:
        Input representing an image.

        Must have suitable format. Some models require tensors to be
        rank 4 in format `(batch_size, dims, ..., channels)` (channels last)
        or `(batch_size, channels, dims, ...)` (channels first),
        where `dims` is usually in order `height, width`
        and `batch_size` is 1 for a single image.

        If ``image`` argument is not given, an image will be created
        from ``doc``, where possible.

    :param image:
        Pillow image over which to overlay the heatmap.
        Corresponds to the input ``doc``.
    :type image: PIL.Image.Image, optional


    See :func:`eli5.keras.explain_prediction.explain_prediction_keras` 
    for a description of ``model``, ``doc``, ``targets``, and ``layer`` parameters.


    Returns
    -------
    expl : eli5.base.Explanation
      An :class:`eli5.base.Explanation` object with the following attributes:
          * ``image`` a Pillow image representing the input.
          * ``targets`` a list of :class:`eli5.base.TargetExplanation` objects \
              for each target. Currently only 1 target is supported.
      The :class:`eli5.base.TargetExplanation` objects will have the following attributes:
          * ``heatmap`` a rank 2 numpy array with the localization map \
            values as floats.
          * ``target`` ID of target class.
          * ``score`` value for predicted class.
    """
    if image is None:
        image = _extract_image(doc)
    _validate_doc(model, doc)
    activation_layer = _get_activation_layer(model, layer)

    # TODO: maybe do the sum / loss calculation in this function and pass it to gradcam.
    # This would be consistent with what is done in
    # https://github.com/ramprs/grad-cam/blob/master/misc/utils.lua
    # and https://github.com/ramprs/grad-cam/blob/master/classification.lua
    values = gradcam_backend(model, doc, targets, activation_layer)
    weights, activations, grads, predicted_idx, predicted_val = values
    heatmap = gradcam(weights, activations)

    return Explanation(
        model.name,
        description=DESCRIPTION_KERAS,
        error='',
        method='Grad-CAM',
        image=image,
        targets=[TargetExplanation(
            predicted_idx,
            score=predicted_val, # for now we keep the prediction in the .score field (not .proba)
            heatmap=heatmap, # 2D [0, 1] numpy array
        )],
        is_regression=False, # might be relevant later when explaining for regression tasks
        highlight_spaces=None, # might be relevant later when explaining text models
    )


def _maybe_image(model, doc):
    # type: (Model, np.ndarray) -> bool
    """Decide whether we are dealing with a image-based explanation 
    based on heuristics on ``model`` and ``doc``."""
    return _maybe_image_input(doc) and _maybe_image_model(model)


def _maybe_image_input(doc):
    # type: (np.ndarray) -> bool
    """Decide whether ``doc`` represents an image input."""
    rank = len(doc.shape)
    # image with channels or without (spatial only)
    return rank == 4 or rank == 3


def _maybe_image_model(model):
    # type: (Model) -> bool
    """Decide whether ``model`` is used for images."""
    # FIXME: replace try-except with something else
    try:
        # search for the first occurrence of an "image" layer
        _search_layer_backwards(model, _is_possible_image_model_layer)
        return True
    except ValueError:
        return False


image_model_layers = (Conv2D,
                      MaxPooling2D,
                      AveragePooling2D,
                      GlobalMaxPooling2D,
                      GlobalAveragePooling2D,
                      )


def _is_possible_image_model_layer(model, layer):
    # type: (Model, Layer) -> bool
    """Check that the given ``layer`` is usually used for images."""
    return isinstance(layer, image_model_layers)


def _extract_image(doc):
    # type: (np.ndarray) -> 'PIL.Image.Image'
    """Convert ``doc`` tensor to image."""
    im_arr, = doc  # rank 4 batch -> rank 3 single image
    image = array_to_img(im_arr)
    return image


def _validate_doc(model, doc):
    # type: (Model, np.ndarray) -> None
    """
    Check that the input ``doc`` is suitable for ``model``.
    """
    if not isinstance(doc, np.ndarray):
        raise TypeError('doc must be a numpy.ndarray, got: {}'.format(doc))
    input_sh = model.input_shape
    doc_sh = doc.shape
    if len(input_sh) == 4:
        # rank 4 with (batch, ...) shape
        # check that we have only one image (batch size 1)
        single_batch = (1,) + input_sh[1:]
        if doc_sh != single_batch:
            raise ValueError('Batch size does not match (must be 1). ' 
                             'doc must be of shape: {}, '
                             'got: {}'.format(single_batch, doc_sh))
    else:
        # other shapes
        if doc_sh != input_sh:
            raise ValueError('Input and doc shapes do not match.'
                             'input: {}, doc: {}'.format(input_sh, doc_sh))


def _get_activation_layer(model, layer):
    # type: (Model, Union[None, int, str, Layer]) -> Layer
    """
    Get an instance of the desired activation layer in ``model``,
    as specified by ``layer``.
    """
    if layer is None:
        # Automatically get the layer if not provided
        activation_layer = _search_layer_backwards(model, _is_suitable_activation_layer)
        return activation_layer

    if isinstance(layer, Layer):
        activation_layer = layer
    # get_layer() performs a bottom-up horizontal graph traversal
    # it can raise ValueError if the layer index / name specified is not found
    elif isinstance(layer, int):
        activation_layer = model.get_layer(index=layer)
    elif isinstance(layer, str):
        activation_layer = model.get_layer(name=layer)
    else:
        raise TypeError('Invalid layer (must be str, int, keras.layers.Layer, or None): %s' % layer)

    if _is_suitable_activation_layer(model, activation_layer):
        # final validation step
        return activation_layer
    else:
        raise ValueError('Can not perform Grad-CAM on the retrieved activation layer')


def _search_layer_backwards(model, condition):
    # type: (Model, Callable[[Model, Layer], bool]) -> Layer
    """
    Search for a layer in ``model``, backwards (starting from the output layer),
    checking if the layer is suitable with the callable ``condition``,
    """
    # linear search in reverse through the flattened layers
    for layer in model.layers[::-1]:
        if condition(model, layer):
            # linear search succeeded
            return layer
    # linear search ended with no results
    raise ValueError('Could not find a suitable target layer automatically.')        


def _is_suitable_activation_layer(model, layer):
    # type: (Model, Layer) -> bool
    """
    Check whether the layer ``layer`` matches what is required 
    by ``model`` to do Grad-CAM on ``layer``.
    Returns a boolean.

    Matching Criteria:
        * Rank of the layer's output tensor.
    """
    # TODO: experiment with this, using many models and images, to find what works best
    # Some ideas: 
    # check layer type, i.e.: isinstance(l, keras.layers.Conv2D)
    # check layer name

    # a check that asks "can we resize this activation layer over the image?"
    rank = len(layer.output_shape)
    required_rank = len(model.input_shape)
    return rank == required_rank
