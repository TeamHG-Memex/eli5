# -*- coding: utf-8 -*-
from typing import Optional, Union, Tuple, List

import numpy as np # type: ignore
from scipy.interpolate import interp1d # type: ignore

from eli5.base import (
    WeightedSpans,
    DocWeightedSpans,
)


def gradcam_text_spans(heatmap, # type: np.ndarray
                       tokens, # type: Union[np.ndarray, list]
                       doc, # type: np.ndarray
                       pad_value=None, # type: Optional[Union[int, float]]
                       pad_token=None, # type: Optional[str]
                       interpolation_kind='linear' # type: Union[str, int]
                       ):
    # type: (...) -> Tuple[Union[np.ndarray, list], np.ndarray, WeightedSpans]
    """
    Create text spans from a Grad-CAM ``heatmap`` imposed over ``tokens``.
    Optionally cut off the padding from the explanation
    with the ``pad_value`` or ``pad_token`` arguments.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Array of weights. May be resized to match the length of tokens.

        **Should be rank 1 (no batch dimension).**

    tokens : numpy.ndarray or list
        Tokens that will be highlighted using weights from ``heatmap``.

    doc: numpy.ndarray
        Original input to the network, from which ``heatmap`` was created.

    pad_value: int or float, optional
        Padding number into ``doc``.

    pad_token: str, optional
        Padding symbol into ``tokens``.

        Pass one of either `pad_value` or `pad_token` to cut off padding.

    interpolation_kind: str or int, optional
        Interpolation method. See :func:`eli5.nn.text.resize_1d` for more details.

    Returns
    -------
    (tokens, heatmap, weighted_spans) : (list or numpy.ndarray, numpy.ndarray, WeightedSpans)
        ``tokens`` and ``heatmap`` optionally cut from padding.
        A :class:`eli5.base.WeightedSpans` object with a weight for each token.
    """
    # FIXME: might want to do this when formatting the explanation?
    # TODO: might want to add validation for heatmap and other arguments?
    length = len(tokens)
    heatmap = resize_1d(heatmap, length, interpolation_kind=interpolation_kind)

    # values will be cut off from the *resized* heatmap
    if pad_value is not None or pad_token is not None:
        # remove padding
        pad_indices = _find_padding(pad_value=pad_value, pad_token=pad_token, doc=doc, tokens=tokens)
        # If pad_value is not the actual padding value, behaviour is unknown
        tokens, heatmap = _trim_padding(pad_indices, tokens, heatmap)
    document = _construct_document(tokens)
    spans = _build_spans(tokens, heatmap, document)
    weighted_spans = WeightedSpans([
        DocWeightedSpans(document, spans=spans)
    ]) # why list? - for each vectorized - don't need multiple vectorizers?
       # multiple highlights? - could do positive and negative expl?
    return tokens, heatmap, weighted_spans


def resize_1d(heatmap, length, interpolation_kind='linear'):
    # type: (np.ndarray, int, Union[str, int]) -> np.ndarray
    """
    Resize the ``heatmap`` 1D array to match the specified ``length``.

    For example, upscale/upsample a heatmap with length 400
    to have length 500.

    Parameters
    ----------

    heatmap : numpy.ndarray
        Heatmap to be resized.

    length : int
        Required width.

    interpolation_kind : str or int, optional
        Interpolation method used by the underlying ``scipy.interpolate.interp1d`` resize function.

        Used when resizing ``heatmap`` to the correct ``length``.

        Default is ``linear``.

    Returns
    -------
    heatmap : numpy.ndarray
        The heatmap resized.
    """
    if len(heatmap.shape) == 1 and heatmap.shape[0] == 1:
        # single weight, no batch
        heatmap = heatmap.repeat(length)
    else:
        # more than length 1

        # scipy.interpolate solution
        # https://stackoverflow.com/questions/29085268/resample-a-numpy-array
        # interp1d requires at least length 2 array
        y = heatmap  # data to interpolate
        x = np.linspace(0, 1, heatmap.size) # array matching y
        interpolant = interp1d(x, y, kind=interpolation_kind) # approximates y = f(x)
        z = np.linspace(0, 1, length)  # points where to interpolate
        heatmap = interpolant(z)  # interpolation result

        # other solutions include scipy.signal.resample (periodic, so doesn't apply)
        # and Pillow image fromarray with mode 'F'/etc and resizing (didn't seem to work)
    return heatmap


def _build_spans(tokens, # type: Union[np.ndarray, list]
                 heatmap, # type: np.ndarray
                 document, # type: str
                 ):
    """Highlight ``tokens`` in ``document``, with weights from ``heatmap``."""
    assert len(tokens) == len(heatmap)
    spans = []
    running = 0  # where to start looking for token in document
    for (token, weight) in zip(tokens, heatmap):
        # find first occurrence of token, on or after running count
        t_start = document.index(token, running)
        # create span
        t_end = t_start + len(token)
        span = (token, [(t_start, t_end,)], weight)
        spans.append(span)
        # update run
        running = t_end
    return spans


def _construct_document(tokens):
    # type: (Union[list, np.ndarray]) -> str
    """Create a document string by joining ``tokens``."""
    if _is_character_tokenization(tokens):
        sep = ''
    else:
        sep = ' '
    return sep.join(tokens)


def _is_character_tokenization(tokens):
    # type: (Union[list, np.ndarray]) -> bool
    """
    Check whether tokenization is character-level
    (returns True) or word-level (returns False).
    """
    return any(' ' in t for t in tokens)


def _find_padding(pad_value=None, # type: Union[int, float]
                  pad_token=None, # type: str
                  doc=None, # type: Optional[np.ndarray]
                  tokens=None # type: Optional[Union[np.ndarray, list]]
                  ):
    # type: (...) -> np.ndarray
    """Dispatch to a padding finder based on arguments."""
    # check that did not pass both pad_value and pad_token
    # which is ambiguous (which should take precedence?)
    assert pad_value is None or pad_token is None
    if pad_value is not None and doc is not None:
        return _find_padding_values(pad_value, doc)
    elif pad_token is not None and tokens is not None:
        return _find_padding_tokens(pad_token, tokens)
    else:
        raise TypeError('Pass "doc" and "pad_value", '
                        'or "tokens" and "pad_token".')
    # TODO: warn if indices is empty - passed wrong padding char/value?


def _find_padding_values(pad_value, doc):
    # type: (Union[int, float], np.ndarray) -> np.ndarray
    if not isinstance(pad_value, (int, float)):
        raise TypeError('"pad_value" must be int or float. Got "{}"'.format(type(pad_value)))
    values, indices = np.where(doc == pad_value)
    return indices


def _find_padding_tokens(pad_token, tokens):
    # type: (str, Union[list, np.ndarray]) -> np.ndarray
    if not isinstance(pad_token, str):
        raise TypeError('"pad_token" must be str. Got "{}"'.format(type(pad_token)))
    indices = [idx for idx, token in enumerate(tokens) if token == pad_token]
    return np.array(indices)


def _trim_padding(pad_indices, # type: np.ndarray
                  tokens, # type: Union[list, np.ndarray]
                  heatmap, # type: np.ndarray
                  ):
    # type: (...) -> Tuple[Union[list, np.ndarray], np.ndarray]
    """Remove padding from ``tokens`` and ``heatmap``."""
    # heatmap and tokens must be same length?
    if 0 < len(pad_indices):
        # found some padding symbols

        # delete all values along indices
        # this is not as robust as explicitly finding pre and post padding characters
        # and we can not detect and raise an error if there is padding in the middle of the text
        tokens = np.delete(tokens, pad_indices)
        heatmap = np.delete(heatmap, pad_indices)
    return tokens, heatmap