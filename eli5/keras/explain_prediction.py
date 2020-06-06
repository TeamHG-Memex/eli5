# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import Union, Optional, Callable, List, Tuple, Generator, TYPE_CHECKING
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
    Conv1D,
    Embedding,
    AveragePooling1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    RNN,
    LSTM,
    GRU,
    Bidirectional,
)
from keras.preprocessing.image import array_to_img

from eli5.base import (
    Explanation, 
    TargetExplanation, 
)
from eli5.explain import explain_prediction
from eli5.nn.gradcam import (
    gradcam_heatmap,
    DESCRIPTION_GRADCAM,
)
from eli5.nn.text import (
    gradcam_spans,
)
from .gradcam import (
    gradcam_backend_keras,
)


# note that keras.models.Sequential subclasses keras.models.Model
@explain_prediction.register(Model)
def explain_prediction_keras(model, # type: Model
                             doc, # type: np.ndarray
                             targets=None, # type: Optional[list]
                             layer=None, # type: Optional[Union[int, str, Layer]]
                             relu=True, # type: bool
                             counterfactual=False, # type: bool
                             image=None,
                             tokens=None,
                             pad_value=None,
                             pad_token=None,
                             interpolation_kind='linear',
                             ):
    # type: (...) -> Explanation
    """
    Explain the prediction of a Keras classifier with the Grad-CAM technique.

    We explicitly assume that the model's task is classification, i.e. final output is class scores.

    :param keras.models.Model model:
        Instance of a Keras neural network model,
        whose predictions are to be explained.


        :raises ValueError: if ``model`` can not be differentiated.
    :param numpy.ndarray doc:
        An input to ``model`` whose prediction will be explained.

        Currently only numpy arrays are supported.
        Also the only data format supported is "channels last".

        The tensor must be of suitable shape for the ``model``.

        Check ``model.input_shape`` to confirm the required dimensions of the input tensor.


        :raises TypeError: if ``doc`` is not a numpy array.
        :raises ValueError: if ``doc`` batch size is not 1.

    :param targets:
        Prediction ID's to focus on.

        *Currently only the first prediction from the list is explained*.
        The list must be length one.

        If None, the model is fed the input ``doc`` and the top prediction 
        is taken as the target automatically.


        :raises ValueError: if ``targets`` is a list with more than one item.
        :raises TypeError: if ``targets`` is not list or None.
        :raises TypeError: if ``targets`` does not contain an integer target.
        :raises ValueError: if integer target is not in the classes that ``model`` predicts.
    :type targets: list[int], optional

    :param layer:
        The activation layer in the model to perform Grad-CAM on:
        a valid keras layer name, layer index, or an instance of a Keras layer.

        If None, a suitable layer is attempted to be chosen automatically.


        :raises TypeError: if ``layer`` is not None, str, int, or keras.layers.Layer instance.
        :raises ValueError: if suitable layer can not be found automatically.
        :raises ValueError: if differentiation fails with respect to retrieved ``layer``.
    :type layer: int or str or keras.layers.Layer, optional

    :param relu:
        Whether to apply ReLU on the resulting heatmap.

        Set to `False` to see the "negative" of a class.

        Default is `True`.
    :type relu: bool, optional

    :param counterfactual:
        Whether to negate gradients during the heatmap calculation.
        Useful for highlighting what makes the prediction or class score go down.

        Default is `False`.
    :type counterfactual: bool, optional


    See :func:`eli5.explain_prediction` for more information about the ``model``,
    ``doc``, and ``targets`` parameters.


    Other arguments are passed to concrete implementations
    for image and text explanations.


    Returns
    -------
      expl : :class:`eli5.base.Explanation`
        An :class:`eli5.base.Explanation` object for the relevant implementation.

        The following attributes are supported by all concrete implementations:

        * ``targets`` a list of :class:`eli5.base.TargetExplanation` objects \
            for each target. Currently only 1 target is supported.
        * ``layer`` used for Grad-CAM.

    """
    # Note that this function should only do dispatch 
    # and no other processing

    # check that only one of image or tokens is passed
    assert image is None or tokens is None
    if image is not None or _maybe_image(model, doc):
        return explain_prediction_keras_image(model,
                                              doc,
                                              image=image,
                                              targets=targets,
                                              layer=layer,
                                              relu=relu,
                                              counterfactual=counterfactual,
                                              )
    elif tokens is not None:
        return explain_prediction_keras_text(model,
                                             doc,
                                             tokens=tokens,
                                             pad_value=pad_value,
                                             pad_token=pad_token,
                                             interpolation_kind=interpolation_kind,
                                             targets=targets,
                                             layer=layer,
                                             relu=relu,
                                             counterfactual=counterfactual,
                                             )
    else:
        return explain_prediction_keras_not_supported(model, doc)


# Some parameters to explain_prediction_keras* functions are repeated
# Watch that the default values match


def explain_prediction_keras_not_supported(model, doc):
    """
    Can not do an explanation based on the passed arguments.
    Did you pass either "image" or "tokens"?
    """
    # An alternative to giving up is to still generate a heatmap
    # but not do any formatting, i.e. useful if we do not support some task.
    return Explanation(
        model.name,
        error='model "{}" is not supported, '
              'try passing the "image" argument if explaining an image model, '
              'or the "tokens" argument if explaining a text model.'.format(model.name),
    )


def explain_prediction_keras_image(model,
                                   doc,
                                   image=None, # type: Optional['PIL.Image.Image']
                                   targets=None,
                                   layer=None,
                                   relu=True,
                                   counterfactual=False,
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
          * ``targets`` and ``layer`` attributes:
            See :func:`eli5.keras.explain_prediction.explain_prediction_keras`.

      The :class:`eli5.base.TargetExplanation` objects will have the following attributes:
          * ``heatmap`` a rank 2 numpy array with the localization map \
            values as floats.
          * ``target`` ID of target class.
          * ``score`` value for predicted class.
    """
    _validate_params(model, doc)
    if image is None:
        image = _extract_image(doc)

    if layer is not None:
        activation_layer = _get_layer(model, layer)
    else:
        activation_layer = _autoget_layer_image(model)

    vals = gradcam_backend_keras(model, doc, targets, activation_layer)
    activations, grads, predicted_idx, predicted_val = vals
    heatmap = gradcam_heatmap(activations,
                              grads,
                              relu=relu,
                              counterfactual=counterfactual,
                              )
    # take from batch
    predicted_idx, = predicted_idx
    predicted_val, = predicted_val
    heatmap, = heatmap

    return Explanation(
        model.name,
        description=DESCRIPTION_GRADCAM,
        error='',
        method='Grad-CAM',
        image=image, # RGBA Pillow image
        layer=activation_layer,
        targets=[TargetExplanation(
            predicted_idx,
            score=predicted_val, # for now we keep the prediction in the .score field (not .proba)
            heatmap=heatmap, # 2D numpy array
        )],
        is_regression=False, # might be relevant later when explaining for regression tasks
    )


def explain_prediction_keras_text(model,
                                  doc,
                                  tokens=None, # type: Optional[Union[List[str], np.ndarray]]
                                  pad_value=None, # type: Optional[Union[int, float]]
                                  pad_token=None, # type: Optional[str]
                                  interpolation_kind='linear', # type: Union[str, int]
                                  targets=None,
                                  layer=None,
                                  relu=True,
                                  counterfactual=False,
                                  ):
    """
    Explain a text-based model, highlighting parts of text that contributed to the prediction.

    In the case of binary classification, this highlights what makes the output go up.

    See :func:`eli5.keras.explain_prediction.explain_prediction_keras` for description of ``targets``, 
    ``layer``, ``relu``, and ``counterfactual`` parameters.

    :param numpy.ndarray doc:
        Suitable input tensor. Temporal with batch size. May have padding.

    :param tokens:
        Tokens that correspond to ``doc``.
        With padding if ``doc`` has padding.

        A Python list or a numpy array of strings. With the same length as ``doc``.
        If ``doc`` has batch size = 1, batch dimension from tokens may be omitted.

        These tokens will be highlighted for text-based explanations.


        :raises TypeError: if ``tokens`` has wrong type or contains wrong types.
        :raises ValueError: if ``tokens`` dimensions do not match.
    :type tokens: list[str], optional

    :param pad_value:
        Integer identifier of the padding token.

        If given, cuts padding off.
    :type pad_value: int or float, optional

    :param pad_token:
        A string token inside ``tokens`` identifying padding.

        If given, cuts padding off.
    :type pad_token: str, optional

    :param interpolation_kind:
        Interpolation method. See :func:`eli5.nn.text.resize_1d` for more details.
    :type interpolation_kind: str or int, optional

    Returns
    -------
    expl : eli5.base.Explanation
      An :class:`eli5.base.Explanation` object with the following attributes:
          * ``targets`` and ``layer`` attributes:
            See :func:`eli5.keras.explain_prediction.explain_prediction_keras`.

      The :class:`eli5.base.TargetExplanation` objects will have the following attributes:
          * ``weighted_spans`` a :class:`eli5.base.WeightedSpans` object with \
            weights for parts of text to be highlighted.
          * ``heatmap`` a rank 1 numpy array with with the localization map \
              values as floats.
          * ``target`` ID of target class.
          * ``score`` value for predicted class.

    """
    assert tokens is not None
    _validate_params(model, doc)
    tokens = _unbatch_tokens(tokens)

    if layer is not None:
        activation_layer = _get_layer(model, layer)
    else:
        activation_layer = _autoget_layer_text(model)

    vals = gradcam_backend_keras(model, doc, targets, activation_layer)
    activations, grads, predicted_idx, predicted_val = vals
    heatmap = gradcam_heatmap(activations,
                              grads,
                              relu=relu,
                              counterfactual=counterfactual,
                              )
    # take from batch
    predicted_idx, = predicted_idx
    predicted_val, = predicted_val
    heatmap, = heatmap
    text_vals = gradcam_spans(heatmap,
                              tokens,
                              doc,
                              pad_value=pad_value,
                              pad_token=pad_token,
                              interpolation_kind=interpolation_kind,
                              )
    # TODO: padding could be relevant for images too?
    tokens, heatmap, weighted_spans = text_vals
    return Explanation(
        model.name,
        description=DESCRIPTION_GRADCAM,
        error='',
        method='Grad-CAM',
        layer=activation_layer,
        targets=[TargetExplanation(
            predicted_idx,
            weighted_spans=weighted_spans,
            score=predicted_val,
            heatmap=heatmap,  # 1D numpy array
        )],
        is_regression=False,  # might be relevant later when explaining for regression tasks
    )
    # TODO: might want to set feature_weights so that you can see the prediction and score
    # as in https://eli5.readthedocs.io/en/latest/tutorials/sklearn-text.html


def _maybe_image(model, doc):
    # type: (Model, np.ndarray) -> bool
    """
    Decide whether we are dealing with a image-based explanation
    based on heuristics on ``model`` and ``doc``.
    """
    return _maybe_image_input(doc) and _maybe_image_model(model)


def _maybe_image_input(doc):
    # type: (np.ndarray) -> bool
    """Decide whether ``doc`` represents an image input."""
    try:
        _validate_doc(doc)
    except (TypeError, ValueError):
        return False
    else:
        rank = len(doc.shape)
        # image with channels or without (spatial only)
        return rank == 4 or rank == 3


def _maybe_image_model(model):
    # type: (Model) -> bool
    """Decide whether ``model`` is used for images."""
    # search for the first occurrence of an "image" layer
    l = _search_layer(model,
                     _backward_layers,
                     lambda model, layer:
                     isinstance(layer, image_model_layers)
                     )
    return l is not None


image_model_layers = (Conv2D, MaxPooling2D, AveragePooling2D, 
                      GlobalMaxPooling2D, GlobalAveragePooling2D,)


def _extract_image(doc):
    # type: (np.ndarray) -> 'PIL.Image.Image'
    """Convert ``doc`` tensor to image."""
    im_arr, = doc  # rank 4 batch -> rank 3 single image
    image = array_to_img(im_arr)
    return image


def _unbatch_tokens(tokens):
    # type: (np.ndarray) -> np.ndarray
    """If ``tokens`` has batch size, take out the first sample from the batch."""
    an_entry = tokens[0]
    if isinstance(an_entry, str):
        # not batched
        return tokens
    else:
        # batched, return first entry
        return an_entry


def _get_layer(model, layer): 
    # type: (Model, Union[int, str, Layer]) -> Layer
    """
    Wrapper around ``model.get_layer()`` for int, str, or Layer argument``.
    Return a keras Layer instance.
    """
    # currently we don't do any validation on the retrieved layer
    if isinstance(layer, Layer):
        return layer
    elif isinstance(layer, int):
        # keras.get_layer() performs a bottom-up horizontal graph traversal
        # the function raises ValueError if the layer index / name specified is not found
        return model.get_layer(index=layer)
    elif isinstance(layer, str):
        return model.get_layer(name=layer)
    else:
        raise TypeError('Invalid layer (must be str, int, or keras.layers.Layer): %s' % layer)


# Heuristics for getting a suitable activation layer

def _autoget_layer_image(model):
    # type: (Model) -> Layer
    """Try find a suitable layer for image ``model``."""
    # TODO: experiment with this, using many models and images, to find what works best
    # a check that asks "can we resize this activation layer over the image?"
    # and "is this layer used for 2d operations?"
    l = _search_layer(model,
                      _backward_layers,
                      lambda model, layer:
                      (len(layer.output_shape) == len(model.input_shape) and
                       '2d' in layer.__class__.__name__.lower())
                      )

    if l is None:
        raise ValueError('Could not find a suitable image layer automatically. '
                         'Try passing the "layer" argument.')
    else:
        return l


def _autoget_layer_text(model):
    # type: (Model) -> Layer
    """Try find a suitable layer for text ``model``."""
    # search for 'word level' features
    # search categories in sequence: text > 1D > embedding

    def wanted(layers):
        return lambda model, layer: (len(layer.output_shape) == 3 and
                                     isinstance(layer, layers))

    l = _search_layer(model, _backward_layers, wanted(text_layers))
    if l is None:
        l = _search_layer(model, _backward_layers, wanted(temporal_layers))
        if l is None:
            l = _search_layer(model, _backward_layers, wanted((Embedding,)))

    if l is None:
        raise ValueError('Could not find a suitable text layer automatically. '
                         'Try passing the "layer" argument.')
    else:
        return l


text_layers = (Conv1D, RNN, LSTM, GRU, Bidirectional,)
temporal_layers = (AveragePooling1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D,)


def _search_layer(model, # type: Model
                  layers_generator, # type: Callable[[Model], Generator[Layer, None, None]]
                  layer_condition, # type: Callable[[Model, Layer], bool]
                  ):
    # type: (...) -> Optional[Layer]
    """
    Search for a layer in ``model``, iterating through layers in the order specified by
    ``layers_generator``, returning the first layer that matches ``layer_condition``.
    If no layer could be found, return None.
    """
    # linear search in reverse through the flattened layers
    for layer in layers_generator(model):
        if layer_condition(model, layer):
            # linear search succeeded
            return layer
    # linear search ended with no results
    return None


def _backward_layers(model):
    # type: (Model) -> Generator[Layer, None, None]
    """Return layers going from output to input (backwards)."""
    return (model.get_layer(index=i) for i in range(len(model.layers)-1, -1, -1))


def _validate_params(model, # type: Model
                     doc, # type: np.ndarray
                     ):
    # type: (...) -> None
    """Helper for validating explanation function parameters."""
    _validate_model(model)
    _validate_doc(doc)


def _validate_model(model):
    if len(model.layers) == 0:
        # "empty" model
        raise ValueError('Model must have at least 1 layer. '
                         'Got model with layers: "{}"'.format(model.layers))


def _validate_doc(doc):
    # type: (np.ndarray) -> None
    """
    Check that the input ``doc`` has the correct type and values.
    """
    if not isinstance(doc, np.ndarray):
        raise TypeError('"doc" must be an instace of numpy.ndarray. '
                        'Got: {} (type "{}")'.format(doc, type(doc)))

    # check that batch=1 (batch greater than 1 is currently not supported)
    batch_size = doc.shape[0]
    if batch_size != 1:
        raise ValueError('"doc" batch size must be 1. '
                         'Got doc with batch size: %d' % batch_size)

    # Note that validation of the input shape, etc is done by Keras