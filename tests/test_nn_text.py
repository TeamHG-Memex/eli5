# -*- coding: utf-8 -*-

import pytest
import numpy as np

from eli5.nn.text import (
    _get_temporal_length,
    resize_1d,
    _build_spans,
    _construct_document,
    _find_padding,
    _trim_padding,
)


@pytest.mark.parametrize('tokens, expected_length', [
    (np.array([[0, 1, 2]]), 3), # batch
    (np.array([0, 1, 2, 3]), 4), # array with no batch
    ([0, 1], 2),
])
def test_get_temporal_length(tokens, expected_length):
    assert _get_temporal_length(tokens) == expected_length


# TODO: test resize_1d with a single element - should be it repeated
@pytest.mark.parametrize('heatmap, width, expected', [
    (np.array([0]), 3, np.array([0, 0, 0])),
    # (np.array([0, 1]), 3, np.array([0, 0.5, 1])),
])
def test_resize_1d(heatmap, width, expected):
    resized = resize_1d(heatmap, width)
    assert np.array_equal(resized, expected)


@pytest.mark.parametrize('tokens, heatmap, document, expected_spans', [
    (['good', 'and', 'bad'], np.array([1, 0, -1]), 'good and bad',
                                                   #012345678901
        [('good', [(0, 4)], 1), ('and', [(5, 8)], 0), ('bad', [(9, 12)], -1)]),
])
def test_build_spans(tokens, heatmap, document, expected_spans):
    spans = _build_spans(tokens, heatmap, document)
    assert len(spans) == len(expected_spans)
    for span, expected_span in zip(spans, expected_spans):
        assert span == expected_span


@pytest.mark.parametrize('tokens, expected_document', [
    (['hello', 'there'], 'hello there'), # token-based
    (['a', ' ', 'c', 'a', 't'], 'a cat'), # char-based
])
def test_construct_document(tokens, expected_document):
    document = _construct_document(tokens)
    assert document == expected_document


@pytest.mark.parametrize('pad_value, doc, tokens, expected_indices', [
    ('<PAD>', None, ['the', 'test', '<PAD>', '<PAD>'], [2, 3]),
    (0, np.array([[0, 0, 1, 2]]), None, np.array([0, 1])),
    # TODO: test float vectors?
])
def test_find_padding(pad_value, doc, tokens, expected_indices):
    indices = _find_padding(pad_value, doc=doc, tokens=tokens)
    np.array_equal(indices, expected_indices)


def test_find_padding_invalid():
    # invalid pad_value
    with pytest.raises(TypeError):
        _find_padding([0, 1], doc=None, tokens=None)
    # invalid combinations
    with pytest.raises(TypeError):
        _find_padding('<PAD>', doc=[0, 2, 1], tokens=None)
    with pytest.raises(TypeError):
        _find_padding(0, doc=None, tokens=['a', 'test'])


@pytest.mark.parametrize('pad_indices, padding, tokens, heatmap, expected_tokens, expected_heatmap', [
    ([0, 1], 'pre', ['PAD', 'PAD', 'hi', 'there'], np.array([0.2, 0.1, 2, 3]), 
        ['hi', 'there'], np.array([2, 3])),
    ([2, 3], 'post', ['hi', 'there', 'PAD', 'PAD'], np.array([2, 3, 0.1, 0.2]),
        ['hi', 'there'], np.array([2, 3]),
    ),
])
def test_trim_padding(pad_indices, padding, tokens, heatmap, expected_tokens, expected_heatmap):
    tokens, heatmap = _trim_padding(pad_indices, padding, tokens, heatmap)
    assert tokens == expected_tokens
    assert np.array_equal(heatmap, expected_heatmap)


# TODO: test_trim_padding with invalid cases


# TODO: test gradcam_text_spans with a small example