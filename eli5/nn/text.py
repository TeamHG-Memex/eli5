# -*- coding: utf-8 -*-
from typing import Optional, Union, Tuple, List

import numpy as np # type: ignore
from scipy.interpolate import interp1d # type: ignore

from eli5.base import (
    WeightedSpans,
    DocWeightedSpans,
)


# TODO: remove gradcam references. Keep this as a text module for neural nets in general??
# FIXME: migth want to move this to nn.gradcam?
def gradcam_text_spans(heatmap, # type: np.ndarray
                       tokens, # type: Union[np.ndarray, list]
                       doc, # type: np.ndarray
                       pad_value=None, # type: Optional[Union[str, int, float]]
                       padding='post', # type: str
                       interpolation_kind='linear' # type: Union[str, int]
    ):
    # type: (...) -> Tuple[Union[np.ndarray, list], np.ndarray, WeightedSpans]
    """
    Create text spans from a Grad-CAM ``heatmap`` imposed over ``tokens``.
    Optionally cut off the padding from the explanation 
    with the ``pad_value`` and ``padding`` arguments.
    """
    # we resize before cutting off padding?
    # FIXME: might want to do this when formatting the explanation?
    width = _get_temporal_length(tokens)
    heatmap = resize_1d(heatmap, width, interpolation_kind=interpolation_kind)

    if pad_value is not None:
        # remove padding
        pad_indices = _find_padding(pad_value, doc, tokens)
        # If pad_value is not the actual padding value, behaviour is unknown
        tokens, heatmap = _trim_padding(pad_indices, padding,
                                        tokens, heatmap)
    document = _construct_document(tokens)
    spans = _build_spans(tokens, heatmap, document)
    weighted_spans = WeightedSpans([
        DocWeightedSpans(document, spans=spans)
    ]) # why list? - for each vectorized - don't need multiple vectorizers?
       # multiple highlights? - could do positive and negative expl?
    return tokens, heatmap, weighted_spans


def _get_temporal_length(tokens):
    # type: (Union[np.ndarray, list]) -> int
    if isinstance(tokens, np.ndarray):
        if len(tokens.shape) == 1:
            # no batch size
            return tokens.shape[0]
        elif len(tokens.shape) == 2:
            # possible batch size
            return tokens.shape[1]
        else:
            raise ValueError('"tokens" shape is ambiguous.')
    elif isinstance(tokens, list):
        return len(tokens)
    else:
        raise TypeError('"tokens" must be an instance of list or numpy.ndarray. '
                        'Got: {}'.format(tokens))


def resize_1d(heatmap, width, interpolation_kind='linear'):
    # type: (np.ndarray, int, Union[str, int]) -> np.ndarray
    """
    Resize heatmap 1D array to match the specified ``width``.
    
    For example, upscale/upsample a heatmap with length 400
    to have length 500.
    """
    if len(heatmap.shape) == 1 and heatmap.shape[0] == 1:
        # single weight, no batch
        heatmap = heatmap.repeat(width)
    else:
        # more than length 1

        # scipy.interpolate solution
        # https://stackoverflow.com/questions/29085268/resample-a-numpy-array
        # interp1d requires at least length 2 array
        y = heatmap # data to interpolate
        x = np.linspace(0, 1, heatmap.size) # array matching y
        interpolant = interp1d(x, y, kind=interpolation_kind) # approximates y = f(x)
        z = np.linspace(0, 1, width) # points where to interpolate
        heatmap = interpolant(z) # interpolation result

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
    running = 0 # where to start looking for token in document
    for (token, weight) in zip(tokens, heatmap): # FIXME: weight can be renamed?
        # find first occurrence of token, on or after running count
        t_start = document.index(token, running)
        # create span
        t_end = t_start + len(token)
        # why start and end is list of tuples?
        span = tuple([token, [(t_start, t_end,)], weight])
        spans.append(span)
        # update run
        running = t_end 
    return spans


def _construct_document(tokens):
    # type: (Union[list, np.ndarray]) -> str
    """Create a document string by joining ``tokens``."""
    if ' ' in tokens:
        # character-based (probably)
        sep = ''
    else:
        # word-based
        sep = ' '
    return sep.join(tokens)


def _find_padding(pad_value, doc=None, tokens=None):
    # (Union[str, int], Optional[np.ndarray], Optional[np.ndarray, list]) -> np.ndarray
    """Find padding in input ``doc`` or ``tokens`` based on ``pad_value``,
    returning a numpy array of indices where padding was found."""
    if isinstance(pad_value, (int, float)) and doc is not None:
        indices = _find_doc_padding(pad_value, doc)
    elif isinstance(pad_value, str) and tokens is not None:
        indices = _find_tokens_padding(pad_value, tokens)
    else:
        raise TypeError('Pass "doc" and "pad_value" as int or float, '
                        'or "tokens" and "pad_value" as str. '
                        'Got: {}'.format(pad_value))
    return indices
    # TODO: warn if indices is empty - passed wrong padding char/value?


def _find_doc_padding(pad_value, doc):
    # type: (int, np.ndarray) -> np.ndarray
    values, indices = np.where(doc == pad_value)
    return indices


def _find_tokens_padding(pad_value, tokens):
    # type: (str, Union[list, np.ndarray]) -> np.ndarray
    indices = [idx for idx, token in enumerate(tokens) if token == pad_value]
    return np.array(indices)


def _trim_padding(pad_indices, # type: np.ndarray
                  padding, # type: str
                  tokens, # type: Union[list, np.ndarray]
                  heatmap, # type: np.ndarray
    ): 
    # type: (...) -> Tuple[Union[list, np.ndarray], np.ndarray]
    """Removing padding from ``tokens`` and ``heatmap``."""
    # heatmap and tokens must be same length?
    if 0 < len(pad_indices):
        # found some padding characters
        if padding == 'post':
            # `word word pad pad` -> 'word word'
            first_pad_idx = pad_indices[0]
            tokens = tokens[:first_pad_idx]
            heatmap = heatmap[:first_pad_idx]
        elif padding == 'pre':
            # `pad pad word word` -> 'word word'
            last_pad_idx = pad_indices[-1]
            tokens = tokens[last_pad_idx+1:]
            heatmap = heatmap[last_pad_idx+1:]
        else:
            raise ValueError('padding must be "post" or "pre". '
                             'Got: {}'.format(padding))
    # TODO: check that there's no padding characters inside the text
    return tokens, heatmap