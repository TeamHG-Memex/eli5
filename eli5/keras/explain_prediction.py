# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import Union, Optional, Callable, List, Tuple, Generator, TYPE_CHECKING

import numpy as np # type: ignore
import keras # type: ignore
import keras.backend as K # type: ignore
from keras.models import Model # type: ignore
from keras.layers import Layer # type: ignore

from eli5.base import (
    Explanation, 
    TargetExplanation, 
)
from eli5.explain import explain_prediction
from eli5.nn.gradcam import (
    gradcam_heatmap,
    DESCRIPTION_GRADCAM
)
from eli5.nn.text import (
    gradcam_text_spans,
    _get_temporal_length,
)
from .gradcam import (
    gradcam_backend_keras,
)


if TYPE_CHECKING:
    import PIL # type: ignore
    # https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING
    # https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
    # Question: do we need to check types of things we ignore?
    # is mypy good for "type documentation"
    # or is it the opposite? (needs maintenance)


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
                             padding='post',
                             interpolation_kind='linear',
                             ):
    # type: (...) -> Explanation
    """
    Explain the prediction of a Keras classifier with the Grad-CAM technique.
    
    # TODO: explain Grad-CAM

    We explicitly assume that the model's task is classification, i.e. final output is class scores.
    
    :param keras.models.Model model:
        Instance of a Keras neural network model, 
        whose predictions are to be explained.


        :raises ValueError: if ``model`` can not be differentiated.

    :param numpy.ndarray doc:
        An input image as a tensor to ``model``, 
        from which prediction will be done and explained.

        Currently only numpy arrays are supported.
        Also the only data format supported is "channels last".

        The tensor must be of suitable shape for the ``model``. 

        For example, some models require input images to be 
        rank 4 in format `(batch_size, dims, ..., channels)` (channels last)
        or `(batch_size, channels, dims, ...)` (channels first), 
        where `dims` is usually in order `height, width`
        and `batch_size` is 1 for a single image.

        Check ``model.input_shape`` to confirm the required dimensions of the input tensor.


        :raises TypeError: if ``doc`` is not a numpy array.
        :raises ValueError: if ``doc`` shape does not match.

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
        
        If None, a suitable layer is attempted to be retrieved. 
        
        For best results, pick a layer that:
            * has spatial or temporal information (conv, recurrent, pool, embedding)
              (not dense layers).
            * shows high level features.
            * has large enough dimensions for resizing over input to work.


        :raises TypeError: if ``layer`` is not None, str, int, or keras.layers.Layer instance.
        :raises ValueError: if suitable layer can not be found.
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
    expl : eli5.base.Explanation
        An :class:`eli5.base.Explanation` object with the following attributes set (some inside ``targets``)
            * ``heatmap`` a numpy array with the localization map values.
            * ``target`` ID of target class.
            * ``score`` value for predicted class.
    """
    # Note that this function should only do dispatch 
    # and no other processing

    # check that only one of image or tokens is passed
    assert image is None or tokens is None
    if image is not None:
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
                                             padding=padding,
                                             interpolation_kind=interpolation_kind,
                                             targets=targets,
                                             layer=layer,
                                             relu=relu,
                                             counterfactual=counterfactual,
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
              'missing "image" or "tokens" argument.'.format(model.name),
    )
    # TODO (open issue): implement 'other'/differentiable network type explanations


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

    See :func:`eli5.keras.explain_prediction.explain_prediction_keras` 
    for a description of ``targets``, ``layer``, ``relu``, and ``counterfactual`` parameters.

    :param image:
        Pillow image over which to overlay the heatmap.

        Corresponds to the input ``doc``.

        Must have mode 'RGBA'.
    :type image: PIL.Image.Image, optional

    Returns
    -------
    expl : eli5.base.Explanation
        An :class:`eli5.base.Explanation` object containing the following additional attributes
            * ``image`` the original Pillow image with mode RGBA.
            * ``heatmap`` rank 2 (2D) numpy array.
    """
    # TODO (open issue): support taking images that are not 'RGBA' -> 'RGB' as well (happens with keras load_img)
    # and grayscale too
    assert image is not None
    _validate_doc(model, doc)

    if layer is not None:
        activation_layer = _get_layer(model, layer)
    else:
        activation_layer = _search_activation_layer(model, 
            _backward_layers, _is_suitable_image_layer)

    activations, grads, predicted_idx, predicted_val = gradcam_backend_keras(model, 
                                            doc, targets, activation_layer)
    heatmap = gradcam_heatmap(activations,
                              grads,
                              relu=relu,
                              counterfactual=counterfactual,
    )
    heatmap, = heatmap # FIXME: hardcode batch=1 for now

    # TODO (open issue): image padding cut off. pass 2-tuple?
    return Explanation(
        model.name,
        description=DESCRIPTION_GRADCAM,
        error='',
        method='Grad-CAM',
        image=image, # RGBA Pillow image
        # PR FIXME: Would be good to include retrieved layer as an attribute
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
                                  pad_value=None, # type: Optional[Union[int, float, str]]
                                  padding='post', # type: str
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

    :param tokens:
        Tokens that correspond to ``doc``.
        With padding if ``doc`` has padding.

        A Python list or a numpy array of strings. With the same length as ``doc``.
        If ``doc`` has batch size = 1, batch dimension from tokens may be omitted.

        These tokens will be highlighted for text-based explanations.
    :type tokens: list[str], optional

    :param pad_value:
        Character for padding. If given, cuts padding off.

        Either an integer value in ``doc``, or a string token in ``tokens``.

        Do not pass this to see the effect of padding on the prediction
        (explain padding).
    :type pad_value: int or str, optional

    :param padding:
        Padding position, either 'pre' (before sequence)
        or 'post' (after sequence).

        Default: 'post'.

        Padding characters will be cut off from the heatmap and tokens.
    :type padding: str, optional

    :param interpolation_kind:
        scipy interpolation to use when resizing the 1D heatmap array.
        Default: 'linear'.
    :type interpolation_kind: str or int, optional

    Returns
    -------
    expl : eli5.base.Explanation
        An :class:`eli5.base.Explanation` object containing the following additional attributes
            * ``weighted_spans`` a :class:`eli5.base.WeightedSpans` object
                with weights for parts of text to be highlighted.
            * ``heatmap`` rank 1 (1D) numpy array.
    """
    # TODO (open issue): implement document vectorizer
    #  :param document:
    #    Full text document for highlighting. 
    #    Not tokenized and without padding.
    # :type document: str, optional
    assert tokens is not None
    _validate_doc(model, doc) # should validate that doc is 2D array (temporal/series data?)
    _validate_tokens(doc, tokens)
    tokens = _unbatch_tokens(tokens)

    if layer is not None:
        activation_layer = _get_layer(model, layer)
    else:
        activation_layer = _search_activation_layer(model, 
            _forward_layers, _is_suitable_text_layer)

    activations, grads, predicted_idx, predicted_val = gradcam_backend_keras(model, 
                                            doc, targets, activation_layer)
    heatmap = gradcam_heatmap(activations,
                              grads,
                              relu=relu,
                              counterfactual=counterfactual,
                              )
    heatmap, = heatmap
    tokens, heatmap, weighted_spans = gradcam_text_spans(heatmap,
                                        tokens,
                                        doc,
                                        pad_value=pad_value,
                                        padding=padding,
                                        interpolation_kind=interpolation_kind,
                                        )

    # FIXME: highlighting is a bit off, eg: all green if is the 0.2 only value in heatmap
    # constrain heatmap in [0, 1] or [-1, 1] and get highlighting to do the same for best results?
    return Explanation(
        model.name,
        description=DESCRIPTION_GRADCAM,
        error='',
        method='Grad-CAM',
        targets=[TargetExplanation(
            predicted_idx,
            weighted_spans=weighted_spans,
            score=predicted_val,
            heatmap=heatmap, # 1D numpy array
        )],
        is_regression=False, # might be relevant later when explaining for regression tasks
        highlight_spaces=None, # might be relevant later when explaining text models
        # TODO: 'preserve_density' argument for char-based highlighting
    )


# There is a problem with repeated arguments to the explain_prediction_keras* functions
# Some arguments are shared, some are unique to each "concrete" explain_prediction*
# An attempt was to make use of **kwargs
# But it led to statements like arg = kwargs.get('arg', None)
# When making changes to argument lists, watch the following:
# * What arguments the "dispatcher" takes?
# * With what arguments the "dispatcher" calls the "concrete" functions?
# * What arguments the "concrete" functions take?
# * Do default values for repeated arguments match?
# If you have a better solution, send a PR / open an issue on GitHub.


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


def _search_activation_layer(model, # type: Model
                             layers_generator, # type: Callable[[Model], Generator[Layer, None, None]]
                             layer_condition, # type: Callable[[Model, Layer], bool]
    ):
    # type: (...) -> Layer
    """
    Search for a layer in ``model``, iterating through layers in the order specified by
    ``layers_generator``, returning the first layer that matches ``layer_condition``.
    """
    # TODO: separate search for image and text - best-results based, not just simple lsearch
    # linear search in reverse through the flattened layers
    for layer in layers_generator(model):
        if layer_condition(model, layer):
            # linear search succeeded
            return layer
    # linear search ended with no results
    raise ValueError('Could not find a suitable target layer automatically.')        


def _forward_layers(model):
    # type: (Model) -> Generator[Layer, None, None]
    """Return layers going from input to output."""
    return (model.get_layer(index=i) for i in range(0, len(model.layers), 1))


def _backward_layers(model):
    # type: (Model) -> Generator[Layer, None, None]
    """Return layers going from output to input (backwards)."""
    return (model.get_layer(index=i) for i in range(len(model.layers)-1, -1, -1))


def _is_suitable_image_layer(model, layer):
    # type: (Model, Layer) -> bool
    """Check whether the layer ``layer`` matches what is required
    by ``model`` to do Grad-CAM on ``layer``, for image-based models.

    Matching Criteria:
    * Rank of the layer's output tensor.
    """
    # TODO: experiment with this, using many models and images, to find what works best
    # Some ideas: 
    # check layer type, i.e.: isinstance(l, keras.layers.Conv2D)
    # check layer name
    # input wrpt output

    # a check that asks "can we resize this activation layer over the image?"
    rank = len(layer.output_shape)
    required_rank = len(model.input_shape)
    return rank == required_rank


def _is_suitable_text_layer(model, layer):
    # type: (Model, Layer) -> bool
    """Check whether ``layer`` is suitable for
    ``model`` to do Grad-CAM on, for text-based models.
    """
    # check the type
    # FIXME: this is not an exhaustive list
    # FIXME: optimisation - this tuple is defined on each call
    desired_layers = (keras.layers.Conv1D,
                      keras.layers.RNN,
                      keras.layers.LSTM,
                      keras.layers.GRU, # TODO: test this
                      keras.layers.Embedding,
                      keras.layers.MaxPooling1D,
                      keras.layers.AveragePooling1D,
    )
    return isinstance(layer, desired_layers) 


def _search_layer_image(model):
    raise NotImplementedError


def _search_layer_text(model):
    raise NotImplementedError


def _validate_doc(model, doc):
    # type: (Model, np.ndarray) -> None
    """
    Check that the input ``doc`` is suitable for ``model``.
    """
    # FIXME: is this validation worth it? Just use Keras validation?
    # Do we make any extra assumptions about doc?
    # https://github.com/keras-team/keras/issues/1641
    # https://github.com/TeamHG-Memex/eli5/pull/315#discussion_r292402171
    # (later we should be able to take tf / backend tensors)
    if not isinstance(doc, np.ndarray):
        raise TypeError('"doc" must be an instace of numpy.ndarray. ' 
                        'Got: {}'.format(doc))
    doc_sh = doc.shape
    batch_size = doc_sh[0]

    # check maching dims
    input_sh = model.input_shape
    if not _eq_shapes(input_sh, doc_sh):
        raise ValueError('"doc" must have shape: {}. '
                         'Got: {}'.format(input_sh, doc_sh))

    # check that batch=1 (will be removed later)
    if batch_size != 1:
        raise ValueError('"doc" batch size must be 1. '
                         'Got doc with batch size: %d' % batch_size)


def _eq_shapes(required, other):
    # type: (Tuple[int], Tuple[int]) -> bool
    """
    Check that ``other`` shape satisfies shape of ``required``.

    For example::
        _eq_shapes((None, 20), (1, 20)) # -> True
    """
    if len(required) != len(other):
        # short circuit based on length
        return False
    matching = [(d1 == d2) # check that same number of dims 
            if (d1 is not None) # if required takes a specific shape for a dim (not None)
            else (1 <= d2) # else just check that the other shape has a valid shape for a dim
            for d1, d2 in zip(required, other)]
    return all(matching)


def _validate_tokens(doc, tokens):
    # type: (np.ndarray, Union[np.ndarray, list]) -> None
    """Check that ``tokens`` contains correct items and matches ``doc``."""
    batch_size, doc_len = doc.shape
    if not isinstance(tokens, (list, np.ndarray)):
        # wrong type
        raise TypeError('"tokens" must be list or numpy.ndarray. '
                        'Got "{}".'.format(tokens))

    an_entry = tokens[0]
    if isinstance(an_entry, str):
        # no batch
        if batch_size != 1:
            # doc is batched but tokens is not
            raise ValueError('If passing "tokens" without batch dimension, '
                             '"doc" must have batch size = 1.'
                             'Got "doc" with batch size = %d.' % batch_size)
        tokens_len = len(tokens)
    elif isinstance(an_entry, (list, np.ndarray)):
        # batched
        a_token = an_entry[0]
        if not isinstance(a_token, str):
            # actual contents are not strings
            raise TypeError('"tokens" must contain strings. '
                            'Got {}'.format(a_token))
        # FIXME: this is hard-coded for batch size = 1
        # each sample's length may vary when type is list
        tokens_len = len(an_entry)
    else:
        raise TypeError('"tokens" must be an array of strings, '
                        'or an array of string arrays. '
                        'Got "{}".'.format(tokens))

    if tokens_len != doc_len:
        raise ValueError('"tokens" and "doc" lengths must match. '
                         '"tokens" length: "%d". "doc" length: "%d"'
                         % (tokens_len, doc_len))


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