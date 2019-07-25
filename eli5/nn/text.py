# -*- coding: utf-8 -*-
import numpy as np # type: ignore
from scipy.signal import resample # type: ignore

from eli5.base import (
    WeightedSpans,
    DocWeightedSpans,
)


# TODO: remove gradcam references. Keep this as a text module for neural nets in general??
def gradcam_text_spans(heatmap, tokens, doc, pad_value=None, padding=None):
    """
    Create text spans from a Grad-CAM ``heatmap`` imposed over ``tokens``.
    Optionally cut off the padding from the explanation 
    with the ``pad_value`` and ``padding`` arguments.
    """
    # we resize before cutting off padding?
    # FIXME: might want to do this when formatting the explanation?
    heatmap = resize_1d(heatmap, tokens)

    if pad_value is not None:
        # remove padding
        padding_indices = _find_padding(pad_value, doc, tokens)
        tokens, heatmap = _trim_padding(padding_indices, padding,
                                        tokens, heatmap)
    document = _construct_document(tokens)
    spans = _build_spans(tokens, heatmap, document)
    weighted_spans = WeightedSpans([
        DocWeightedSpans(document, spans=spans)
    ]) # why list? - for each vectorized - don't need multiple vectorizers?
       # multiple highlights? - could do positive and negative expl?
    return tokens, heatmap, weighted_spans


def resize_1d(heatmap, tokens):
    """
    Resize heatmap to match the length of tokens.
    
    For example, upscale/upsample a length=400 heatmap 
    to match length=500 array of tokens.
    """
    width = len(tokens)

    if len(heatmap.shape) == 1 and heatmap.shape[0] == 1:
        # single weight
        # FIXME (with resample func): resizing for single value heatmap, 
        # i.e. final node in sentiment classification?
        # only highlights first token
        # FIXME: this still gives varied highlighting
        heatmap = heatmap.repeat(width)
    else:
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
        # FIXME: resample assumes "signal is periodic"
        heatmap = resample(heatmap, width)

        # gives poor quality
        # import scipy
        # heatmap = scipy.signal.resample_poly(heatmap, width, 1)

        ## other possibilities
        # https://stackoverflow.com/questions/29085268/resample-a-numpy-array - numpy and scipy interpolation
        # https://machinelearningmastery.com/resample-interpolate-time-series-data-python/ - pandas interpolation
    return heatmap


def _build_spans(tokens, heatmap, document):
    """Highlight ``tokens`` in ``document``, with weights from ``heatmap``."""
    assert len(tokens) == len(heatmap)
    spans = []
    running = 0 # where to start looking for token in document
    for (token, weight) in zip(tokens, heatmap): # FIXME: weight can be renamed?
        # find first occurrence of token, on or after running count
        t_start = document.index(token, running)
        # create span
        t_end = t_start + len(token)
        span = tuple([token, [tuple([t_start, t_end])], weight]) # why start and end is list of tuples?
        spans.append(span)
        # update run
        running = t_end 
    return spans


def _construct_document(tokens):
    """Create a document string by joining ``tokens``."""
    if ' ' in tokens:
        # character-based (probably)
        sep = ''
    else:
        # word-based
        sep = ' '
    return sep.join(tokens)


def _find_padding(pad_value, doc, tokens):
    """Find padding tokens or values, returning indices containing padding characters."""
    # If pad_value is not the actual padding value, behaviour is unknown
    if isinstance(pad_value, int):
        # in doc
        values, indices = np.where(doc == pad_value) # -> all positions with padding character
    elif isinstance(pad_value, str):
        # in tokens
        indices = [idx for idx, token in enumerate(tokens) if token == pad_value]
    else:
        raise TypeError('"pad_value" must be int or str. ' 
                        'Got: {}'.format(pad_value))
    return indices
    # TODO: warn if indices is empty - passed wrong padding char/value?


def _trim_padding(pad_indices, padding, tokens, heatmap):
    """Removing padding from ``tokens`` and ``heatmap``."""
    if 0 < len(pad_indices):
        # found some padding characters
        if padding == 'post':
            # `word word pad pad`
            first_pad_idx = pad_indices[0]
            tokens = tokens[:first_pad_idx]
            heatmap = heatmap[:first_pad_idx]
        elif padding == 'pre':
            # `pad pad word word`
            last_pad_idx = pad_indices[-1]
            tokens = tokens[last_pad_idx+1:]
            heatmap = heatmap[last_pad_idx+1:]
        else:
            raise ValueError('padding must be "post" or "pre". '
                             'Got: {}'.format(padding))
    # TODO: check that there's no padding characters inside the text
    return tokens, heatmap