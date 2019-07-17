# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import Union, Optional, Callable, Tuple, List, Generator, TYPE_CHECKING

import numpy as np # type: ignore
from scipy.signal import resample # type: ignore
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
    import PIL # type: ignore
    # FIXME: do we need to check types of things we ignore?
    # is mypy good for "type documentation"
    # or is it the opposite? (needs maintenance)


DESCRIPTION_KERAS = """
Grad-CAM visualization for classification tasks; 
output is explanation object that contains a heatmap.
"""


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
                             document=None,
                             pad_x=None,
                             padding=None,
                            ):
    # type: (...) -> Explanation
    """
    Explain the prediction of a Keras image classifier.

    We make an explicity assumption that the model's task is classification, i.e. final output is class scores.

    See :func:`eli5.explain_prediction` for more information about the ``model``,
    ``doc``, and ``targets`` parameters.
    
    These arguments are shared by image and text explanations.
    
    :param keras.models.Model model:
        Instance of a Keras neural network model, 
        whose predictions are to be explained.

    :param numpy.ndarray doc:
        An input image as a tensor to ``model``, 
        from which prediction will be done and explained.

        Currently only numpy arrays are supported.

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

        If None, the model is fed the input image and its top prediction 
        is taken as the target automatically.


        :raises ValueError: if ``targets`` is a list with more than one item.
        :raises TypeError: if ``targets`` is not list or None.
    :type targets: list[int], optional

    :param layer:
        The activation layer in the model to perform Grad-CAM on:
        a valid keras layer name, layer index, or an instance of a Keras layer.
        
        If None, a suitable layer is attempted to be retrieved. 
        The layer is searched for going backwards from the output layer, 
        checking that the rank of the layer's output 
        equals to the rank of the input.


        :raises TypeError: if ``layer`` is not None, str, int, or keras.layers.Layer instance.
        :raises ValueError: if suitable layer can not be found.
        :raises ValueError: if differentiation fails with respect to retrieved ``layer``. 
    :type layer: int or str or keras.layers.Layer, optional

    :param relu:
        Whether to apply ReLU on the heatmap.
        Set to `False` to see the "negative" of a class.
    :type relu: bool, optional

    :param counterfactual:
        Whether to negate gradients during
        the heatmap calculation.
        Useful for highlighting what makes
        the prediction or class score go down.
    :type counterfactual: bool, optional

    
    Other arguments are passed to the concrete implementations
    for image or text explanations.


    Returns
    -------
    expl : eli5.base.Explanation
        An ``Explanation`` object with the following attributes set (some inside ``targets``)
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
                                      document=document,
                                      pad_x=pad_x,
                                      padding=padding,
                                      targets=targets,
                                      layer=layer,
                                      relu=relu,
                                      counterfactual=counterfactual,
        )
    else:
        return explain_prediction_keras_not_supported(model, doc)


def explain_prediction_keras_not_supported(model,
                                           doc,
                                           ):
    """Can not do an explanation based on the passed arguments."""
    return Explanation(
        model=model.name,
        error='model "{}" is not supported, '
              'missing "image" or "tokens" argument.'.format(model.name),
    )
    # TODO (open issue): implement 'other'/differentiable network type explanations


def explain_prediction_keras_image(model,
                                   doc,
                                   image=None, # type: Optional[PIL.Image.Image]
                                   targets=None,
                                   layer=None,
                                   relu=True,
                                   counterfactual=False,
    ):
    """
    Explain an image-based model.

    See :func:`eli5.explain_prediction_keras` for description of ``targets``, 
    ``layer``, ``relu``, and ``counterfactual`` parameters.

    :param image:
        Image over which to overlay the heatmap.
    :type image: PIL.Image.Image, optional

    Returns
    -------
    expl : eli5.base.Explanation
        An ``Explanation`` object containing the following additional attributes
            * ``image`` the original Pillow image with mode RGBA.
            * ``heatmap`` rank 2 (2D) numpy array.
    """
    # TODO (open issue): support taking images that are not 'RGBA' -> 'RGB' as well (happens with keras load_img)
    # and grayscale too
    _validate_doc(model, doc)

    activation_layer = _get_activation_layer(model, layer, _backward_layers, _is_suitable_image_layer)
    values = _explanation_backend(model,
                                  doc,
                                  targets,
                                  activation_layer,
                                  relu=relu,
                                  counterfactual=counterfactual,
    )
    heatmap, predicted_idx, predicted_val = values

    # TODO (open issue): image padding cut off. pass 2-tuple?
    
    return Explanation(
        model.name,
        description=DESCRIPTION_KERAS,
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
                                  tokens, # type: Optional[List[str]] # TODO: take as list or numpy array
                                  document=None, # type: Optional[str]
                                  pad_x=None, # type: Optional[int]
                                  padding=None, # type: Optional[str]
                                  targets=None,
                                  layer=None,
                                  relu=True,
                                  counterfactual=False,
                                  ):
    """
    Explain a text-based model.

    See :func:`eli5.explain_prediction_keras` for description of ``targets``, 
    ``layer``, ``relu``, and ``counterfactual`` parameters.

    :param tokens:
        List of input tokens that correspond to doc (may be padded).
        Will be highlighted for text-based explanations.
    :type tokens: list[str], optional

    :param pad_x:
        Character for padding.

        *Not supported for images.*
    :type pad_x: int, optional

    :param padding:
        Padding position, 'pre' (before sequence) 
        or 'post' (after sequence).
        
        Padding characters will be cut off from the heatmap and tokens.
    :type padding: str, optional

    Returns
    -------
    expl : eli5.base.Explanation
        An ``Explanation`` object containing the following additional attributes
            * ``weighted_spans`` weights for parts of text to be highlighted.
            * ``heatmap`` rank 1 (1D) numpy array.
    """
    # TODO (open issue): implement document vectorizer
    #  :param document:
    #    Full text document for highlighting. 
    #    Not tokenized and without padding.
    # :type document: str, optional
    _validate_doc(model, doc)

    activation_layer = _get_activation_layer(model, layer, _forward_layers, _is_suitable_text_layer)
    values = _explanation_backend(model,
                                  doc,
                                  targets,
                                  activation_layer,
                                  relu=relu,
                                  counterfactual=counterfactual,
    )
    heatmap, predicted_idx, predicted_val = values

    heatmap = resize_1d(heatmap, tokens) # might want to do this when formatting the explanation?

    if pad_x is not None:
        # cut off padding
        values, indices = np.where(doc == pad_x)
        if padding == 'post':
            # FIXME: error if no padding is found
            pad_idx = indices[0] # leave +1 just to highlight effect of padding?
            tokens = tokens[:pad_idx]
            heatmap = heatmap[:pad_idx]
        # TODO: pre padding
        # TODO: check that there's no padding characters inside the text
    # TODO: later support document as argument
    document = construct_document(tokens)
    spans = build_spans(tokens, heatmap, document)
    weighted_spans = WeightedSpans([
        DocWeightedSpans(document, spans=spans)
    ]) # why list? - for each vectorized - don't need multiple vectorizers?
       # multiple highlights? - could do positive and negative expl?

    return Explanation(
        model.name,
        description=DESCRIPTION_KERAS,
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
    )


# There is a problem with repeated arguments to the explain_prediction_keras* functions
# Some arguments are shared, some are unique to each "concrete" explain_prediction*
# An attempt was to make use of **kwargs
# But it led to statements like arg = kwargs.get('arg', None)
# When making changes to argument lists, watch the following:
# * What arguments the "dispatcher" takes
# * With what arguments the "dispatcher" calls the "concrete" functions
# * What arguments the "concrete" functions take
# * Do default values for repeated arguments match
# If you have a better solution, send a PR / open an issue on GitHub.


def _explanation_backend(model, doc, targets, activation_layer, relu, counterfactual):
    # TODO: maybe do the sum / loss calculation in this function and pass it to gradcam.
    # This would be consistent with what is done in
    # https://github.com/ramprs/grad-cam/blob/master/misc/utils.lua
    # and https://github.com/ramprs/grad-cam/blob/master/classification.lua
    values = gradcam_backend(model, doc, targets, activation_layer)
    activations, grads, predicted_idx, predicted_val = values
    
    if counterfactual:
        # negate grads for a "counterfactual explanation"
        # can equivalently negate ys loss scalar in gradcam_backend
        grads = -grads

    # FIXME: hardcoding for conv layers, i.e. their shapes
    weights = compute_weights(grads)
    heatmap = gradcam(weights, activations, relu=relu)
    return heatmap, predicted_idx, predicted_val


def _get_activation_layer(model, # type: Model
    layer, # type: Union[None, int, str, Layer]
    layers_generator, # type: layers_arg
    condition # type: condition_arg
    ): 
    # type: (...) -> Layer
    """ 
    Get an instance of the desired activation layer in ``model``,
    as specified by ``layer``.
    """
    # PR FIXME: decouple layer search (no arg) vs simple layer retrieval (arg)
    if layer is None:
        # Automatically get the layer if not provided
        # TODO: search forwards for text models
        activation_layer = _search_layer(model, layers_generator, condition)
        return activation_layer

    if isinstance(layer, Layer):
        return layer

    # get_layer() performs a bottom-up horizontal graph traversal
    # it can raise ValueError if the layer index / name specified is not found
    if isinstance(layer, int):
        activation_layer = model.get_layer(index=layer)
    elif isinstance(layer, str):
        activation_layer = model.get_layer(name=layer)
    else:
        raise TypeError('Invalid layer (must be str, int, keras.layers.Layer, or None): %s' % layer)

    # final validation step
    if condition(model, activation_layer):
        return activation_layer
    else:
        # FIXME: this might not be a useful error message, and the method may be flawed
        # search vs. validation
        raise ValueError('Can not perform Grad-CAM on the retrieved activation layer')


layers_arg = Callable[[Model], Generator[Layer, None, None]]
condition_arg = Callable[[Model, int], bool]

def _search_layer(model, layers, condition):
    # type: (Model, layers_arg, condition_arg) -> Layer
    """
    Search for a layer in ``model``, backwards (starting from the output layer),
    checking if the layer is suitable with the callable ``condition``,
    """
    # linear search in reverse through the flattened layers
    for layer in layers(model):
        if condition(model, layer):
            # linear search succeeded
            return layer
    # linear search ended with no results
    raise ValueError('Could not find a suitable target layer automatically.')        


def _forward_layers(model):
    return (model.get_layer(index=i) for i in range(1, len(model.layers), 1))


def _backward_layers(model):
    return (model.get_layer(index=i) for i in range(len(model.layers)-1, -1, -1))


def _is_suitable_image_layer(model, layer):
    # type: (Model, Layer) -> bool
    """Check whether the layer ``layer`` matches what is required 
    by ``model`` to do Grad-CAM on ``layer``.

    Matching Criteria:
    * Rank of the layer's output tensor."""
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
    """Check whether the layer ``layer`` matches what is required 
    by ``model`` to do Grad-CAM on ``layer``.
    """
    # return isinstance(layer, keras.layers.Conv1D) # FIXME
    return True


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
        raise TypeError('doc must be an instace of numpy.ndarray. ' 
                        'Got: {}'.format(doc))
    doc_sh = doc.shape
    batch_size = doc_sh[0]

    # check maching dims
    input_sh = model.input_shape
    if not _eq_shapes(input_sh, doc_sh):
        raise ValueError('doc must have shape: {}. '
                         'Got: {}'.format(input_sh, doc_sh))

    # check that batch=1 (will be removed later)
    if batch_size != 1:
        raise ValueError('doc batch size must be 1. '
                         'Got doc with batch size: %d' % batch_size)


def _eq_shapes(required, other):
    """
    Check that ``other`` shape satisfies shape of ``required``

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


# TODO: move functions to separate text module

def resize_1d(heatmap, tokens):
    """Resize heatmap to match the length of tokens.
    For example, upscale/upsample a (400,) heatmap 
    to match (500,) array of tokens."""
    width = len(tokens)

    # 1. solution with Pillow image resizing
    # import PIL
    # heatmap = np.expand_dims(heatmap, axis=-1)
    # im = PIL.Image.fromarray(heatmap, mode="F")
    # im = im.resize((width, 1), resample=PIL.Image.LANCZOS)
    # heatmap = np.array(im, dtype='float32')
    # heatmap = heatmap[0, ...]

    # 2. solution with scipy signal resampling
    # apparently this is very slow?
    # can also use resample_poly
    # https://docs.scipy.org/doc/scipy-1.3.0/reference/generated/scipy.signal.resample_poly.html#scipy.signal.resample_poly
    heatmap = resample(heatmap, width)

    ## other possibilities
    # https://stackoverflow.com/questions/29085268/resample-a-numpy-array - numpy and scipy interpolation
    # https://machinelearningmastery.com/resample-interpolate-time-series-data-python/ - pandas interpolation

    return heatmap


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