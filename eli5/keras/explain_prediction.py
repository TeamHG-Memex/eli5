# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import Union, Optional, Callable, Tuple, List, Generator, TYPE_CHECKING

import numpy as np # type: ignore
import keras # type: ignore
import keras.backend as K # type: ignore
from keras.models import Model # type: ignore
from keras.layers import Layer # type: ignore

from eli5.base import (
    Explanation, 
    TargetExplanation, 
    WeightedSpans,
    DocWeightedSpans,
)
from eli5.explain import explain_prediction
from .gradcam import gradcam, gradcam_backend, compute_weights


if TYPE_CHECKING:
    import PIL


DESCRIPTION_KERAS = """Grad-CAM visualization for image classification; 
output is explanation object that contains input image 
and heatmap image for a target.
"""

# note that keras.models.Sequential subclasses keras.models.Model
@explain_prediction.register(Model)
def explain_prediction_keras(estimator, # type: Model
                             doc, # type: np.ndarray
                             target_names=None,
                             targets=None, # type: Optional[list]
                             layer=None, # type: Optional[Union[int, str, Layer]]
                             image=None, # type: Optional[PIL.Image.Image]
                             tokens=None, # type: Optional[List[str]]
                             document=None, # type: Optional[str]
                             pad_idx=None, # type: Optional[int]
                             padding=None, # type: Optional[str]
                            ):
    # type: (...) -> Explanation
    """
    Explain the prediction of a Keras image classifier.

    We make two explicit assumptions
        * The input is images.
        * The model's task is classification, i.e. final output is class scores.

    See :func:`eli5.explain_prediction` for more information about the ``estimator``,
    ``doc``, ``target_names``, and ``targets`` parameters.

    
    :param keras.models.Model estimator:
        Instance of a Keras neural network model, 
        whose predictions are to be explained.


    :param numpy.ndarray doc:
        An input image as a tensor to ``estimator``, 
        from which prediction will be done and explained.

        For example a ``numpy.ndarray``.

        The tensor must be of suitable shape for the ``estimator``. 

        For example, some models require input images to be 
        rank 4 in format `(batch_size, dims, ..., channels)` (channels last)
        or `(batch_size, channels, dims, ...)` (channels first), 
        where `dims` is usually in order `height, width`
        and `batch_size` is 1 for a single image.

        Check ``estimator.input_shape`` to confirm the required dimensions of the input tensor.


        :raises TypeError: if ``doc`` is not a numpy array.
        :raises ValueError: if ``doc`` shape does not match.

    :param target_names:         
        *Not Implemented*.
        Names for classes in the final output layer.
    :type target_names: list, optional

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
        The activation layer in the model to perform Grad-CAM on,
        a valid keras layer name, layer index, or an instance of a Keras layer.
        
        If None, a suitable layer is attempted to be retrieved. 
        See :func:`eli5.keras._search_layer` for details.


        :raises TypeError: if ``layer`` is not None, str, int, or keras.layers.Layer instance.
        :raises ValueError: if suitable layer can not be found.
        :raises ValueError: if differentiation fails with respect to retrieved ``layer``. 
    :type layer: int or str or keras.layers.Layer, optional

    :param image:
        Image over which to overlay the heatmap.
    :type image: PIL.Image.Image, optional

    :param tokens:
        List of input tokens that correspond to doc (may be padded).
        Will be highlighted for text-based explanations.
    :type tokens: list[str], optional

    :param document:
       Full text document for highlighting. 
       Not tokenized and without padding.
    :type document: str, optional

    :param pad_idx:
        Starting index for padding. Will be cut off in the heatmap and tokens.

        *Not supported for images.*
    :type pad_idx: int, optional

    :param padding:
        Padding position, 'pre' (before sequence) 
        or 'post' (after sequence).
    :type padding: str, optional


    Returns
    -------
    expl : eli5.base.Explanation
        An ``Explanation`` object that includes the following attributes (some inside ``targets``)
            * ``heatmap`` a numpy array with the localization map values.
            * ``target`` ID of target class.
            * ``proba`` output for target class for ``softmax`` or ``sigmoid`` outputs.
            * ``score`` output for target class for other activations.
    """
    _validate_doc(estimator, doc)

    if image is None and len(doc.shape) == 4:
        # FIXME
        # for back compatibility
        # might not be good - some rank 4 things may not be images!
        # conversion might be more complicated / might fail!
        # automatically try get image from doc
        # specific for images
        # rank 4 batch -> rank 3 single image
        image = keras.preprocessing.image.array_to_img(doc[0]) # -> RGB Pillow image
        image = image.convert(mode='RGBA')

    if image is not None:
        activation_layer = _get_activation_layer(estimator, layer, _backward_layers, _is_suitable_image_layer)
    else:
        activation_layer = _get_activation_layer(estimator, layer, _forward_layers, _is_suitable_text_layer)
    
    # TODO: maybe do the sum / loss calculation in this function and pass it to gradcam.
    # This would be consistent with what is done in
    # https://github.com/ramprs/grad-cam/blob/master/misc/utils.lua
    # and https://github.com/ramprs/grad-cam/blob/master/classification.lua
    values = gradcam_backend(estimator, doc, targets, activation_layer)
    activations, grads, predicted_idx, predicted_val = values
    # grads = -grads # negate for a "counterfactual explanation"
    # FIXME: hardcoding for conv layers, i.e. their shapes
    weights = compute_weights(grads)
    heatmap = gradcam(weights, activations)

    # classify predicted_val as either a probability or a score
    proba = None
    score = None
    if _outputs_proba(estimator):
        proba = predicted_val
    else:
        score = predicted_val 

    # TODO: cut off padding from text
    # what about images? pass 2 tuple?
    if pad_idx is None:
        pass
    else:
        pass

    if image is not None:
        weighted_spans = None
    else:
        if document is None:
            document = construct_document(tokens)
        spans = build_spans(tokens, heatmap, document)
        weighted_spans = WeightedSpans([
            DocWeightedSpans(document, spans=spans)
        ]) # why list? - for each vectorized - don't need multiple vectorizers?
           # multiple highlights? - could do positive and negative expl?

    return Explanation(
        estimator.name,
        description=DESCRIPTION_KERAS,
        error='',
        method='Grad-CAM',
        image=image, # RGBA Pillow image
        # PR FIXME: Would be good to include retrieved layer as an attribute
        targets=[TargetExplanation(
            predicted_idx,
            weighted_spans=weighted_spans,
            proba=proba,
            score=score,
            heatmap=heatmap, # 2D [0, 1] numpy array
        )],
        is_regression=False, # might be relevant later when explaining for regression tasks
        highlight_spaces=None, # might be relevant later when explaining text models
    )


def construct_document(tokens):
    return ' '.join(tokens)


def build_spans(tokens, heatmap, document):
    # FIXME: use document arg
    spans = []
    running = 0
    for (token, weight) in zip(tokens, heatmap): # FIXME: weight can be renamed?
        t_len = len(token)
        t_start = running
        t_end = t_start + t_len
        span = tuple([token, [tuple([t_start, t_end])], weight]) # why start and end is list of tuples?
        running = t_end + 1 # exclude space
        # print(N, token, weight, i, j)
        spans.append(span)
    return spans


def explain_prediction_keras_image(estimator,
                                   doc,
                                   image,
                                   target_names=None,
                                   targets=None,
                                   layer=None,
    ):
    pass


def explain_prediction_keras_text(estimator,
                                  doc,
                                  tokens,
                                  target_names=None,
                                  targets=None,
                                  layer=None,
                                  document=None,
                                  pad_idx=None,
                                  ):
    pass


def _validate_doc(estimator, doc):
    # type: (Model, np.ndarray) -> None
    """
    Check that the input ``doc`` is suitable for ``estimator``.
    """
    # FIXME: is this validation worth it? Just use Keras validation?
    # Do we make any extra assumptions about doc?
    # https://github.com/keras-team/keras/issues/1641
    # https://github.com/TeamHG-Memex/eli5/pull/315#discussion_r292402171
    # (later we should be able to take tf / backend tensors)
    if not isinstance(doc, np.ndarray):
        raise TypeError('doc must be an instace of numpy.ndarray. ' 
                        'Got: {}'.format(doc))
    doc_sh = doc.shape
    batch_size = doc_sh[0]

    # check maching dims
    input_sh = estimator.input_shape
    if not eq_shapes(input_sh, doc_sh):
        raise ValueError('doc must have shape: {}. '
                         'Got: {}'.format(input_sh, doc_sh))

    # check that batch=1 (will be removed later)
    if batch_size != 1:
        raise ValueError('doc batch size must be 1. '
                         'Got doc with batch size: %d' % batch_size)


def eq_shapes(required, other):
    """
    Check that ``other`` shape satisfies shape of ``required``

    For example::
        eq_shapes((None, 20), (1, 20)) # -> True
    """
    if len(required) != len(other):
        # short circuit based on length
        return False
    matching = [(d1 == d2) # check that same number of dims 
            if (d1 is not None) # if required takes a specific shape for a dim (not None)
            else (1 <= d2) # else just check that the other shape has a valid shape for a dim
            for d1, d2 in zip(required, other)]
    return all(matching)


def _get_activation_layer(estimator, layer, layers_generator, condition):
    # type: (Model, Union[None, int, str, Layer]) -> Layer   ##### FIXME
    """ 
    Get an instance of the desired activation layer in ``estimator``,
    as specified by ``layer``.
    """
    # PR FIXME: decouple layer search (no arg) vs simple layer retrieval (arg)
    if layer is None:
        # Automatically get the layer if not provided
        # TODO: search forwards for text models
        activation_layer = _search_layer(estimator, layers_generator, condition)
        return activation_layer

    if isinstance(layer, Layer):
        return layer

    # get_layer() performs a bottom-up horizontal graph traversal
    # it can raise ValueError if the layer index / name specified is not found
    if isinstance(layer, int):
        activation_layer = estimator.get_layer(index=layer)
    elif isinstance(layer, str):
        activation_layer = estimator.get_layer(name=layer)
    else:
        raise TypeError('Invalid layer (must be str, int, keras.layers.Layer, or None): %s' % layer)

    return activation_layer
    # if _is_suitable_activation_layer(estimator, activation_layer):
    #     # final validation step
    #     # FIXME: this should not be done for text
    #     # this should be moved out
    #     return activation_layer
    # else:
    #     raise ValueError('Can not perform Grad-CAM on the retrieved activation layer')


def _search_layer(estimator, layers, condition):
    # type: (Model, Generator[Layer, None, None], Callable[[Model, int], bool]) -> Layer ### FIXME
    """
    Search for a layer in ``estimator``, backwards (starting from the output layer),
    checking if the layer is suitable with the callable ``condition``,
    """
    # linear search in reverse through the flattened layers
    for layer in layers(estimator):
        if condition(estimator, layer):
            # linear search succeeded
            return layer
    # linear search ended with no results
    raise ValueError('Could not find a suitable target layer automatically.')        


def _forward_layers(estimator):
    return (estimator.get_layer(index=i) for i in range(1, len(estimator.layers), 1))


def _backward_layers(estimator):
    return (estimator.get_layer(index=i) for i in range(len(estimator.layers)-1, -1, -1))


def _is_suitable_image_layer(estimator, layer):
    """Check whether the layer ``layer`` matches what is required 
    by ``estimator`` to do Grad-CAM on ``layer``.

    Matching Criteria:
    * Rank of the layer's output tensor."""
    # type: (Model, Layer) -> bool
    # TODO: experiment with this, using many models and images, to find what works best
    # Some ideas: 
    # check layer type, i.e.: isinstance(l, keras.layers.Conv2D)
    # check layer name
    # input wrpt output

    # a check that asks "can we resize this activation layer over the image?"
    rank = len(layer.output_shape)
    required_rank = len(estimator.input_shape)
    return rank == required_rank


def _is_suitable_text_layer(estimator, layer):
    """Check whether the layer ``layer`` matches what is required 
    by ``estimator`` to do Grad-CAM on ``layer``.
    """
    # type: (Model, Layer) -> bool
    return isinstance(layer, keras.layers.Conv1D)


def _outputs_proba(estimator):
    # type: (Model) -> bool
    """
    Check whether ``estimator`` gives probabilities as its output.
    """
    output_layer = estimator.get_layer(index=-1)
    # we check if the network's output is put through softmax
    # we assume that only softmax can output 'probabilities'

    try:
        actv = output_layer.activation 
    except AttributeError:
        # output layer does not support activation function
        return False
    else:
        return (actv is keras.activations.softmax or 
                actv is keras.activations.sigmoid)