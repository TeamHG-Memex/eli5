# -*- coding: utf-8 -*-
"""eli5.nn package tests for text helpers."""

import pytest
import numpy as np

from eli5.nn.text import (
    resize_1d,
    _build_spans,
    _construct_document,
    _find_padding,
    _find_padding_values,
    _find_padding_tokens,
    _trim_padding,
)


@pytest.mark.parametrize('heatmap, width, expected', [
    (np.array([0]), 3, np.array([0, 0, 0])),
    (np.array([0, 1]), 3, np.array([0, 0.5, 1])),
])
def test_resize_1d(heatmap, width, expected):
    resized = resize_1d(heatmap, width, interpolation_kind='linear')
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


def test_find_padding_value():
    indices = _find_padding_values(0, np.array([[0, 0, 1, 2]]))
    np.array_equal(indices, np.array([0, 1]))


def test_find_padding_token():
    indices = _find_padding_tokens('<PAD>', ['the', 'test', '<PAD>', '<PAD>'])
    np.array_equal(indices, np.array([2, 3]))


def test_find_padding_invalid():
    # invalid combination
    with pytest.raises(TypeError):
        # pad token and doc
        _find_padding(pad_token='<PAD>', doc=[0, 2, 1], tokens=None)


@pytest.mark.parametrize('pad_indices, padding, tokens, heatmap, expected_tokens, expected_heatmap', [
    ([0, 1], 'pre', ['PAD', 'PAD', 'hi', 'there'], np.array([0.2, 0.1, 2, 3]),
        ['hi', 'there'], np.array([2, 3])),
    ([2, 3], 'post', ['hi', 'there', 'PAD', 'PAD'], np.array([2, 3, 0.1, 0.2]),
        ['hi', 'there'], np.array([2, 3])),
])
def test_trim_padding(pad_indices, padding, tokens, heatmap, expected_tokens, expected_heatmap):
    tokens, heatmap = _trim_padding(pad_indices, padding, tokens, heatmap)
    assert tokens == expected_tokens
    assert np.array_equal(heatmap, expected_heatmap)


def test_trim_padding_invalid():
    with pytest.raises(ValueError):
        # currently no such 'padding' side supported
        _trim_padding([1], 'inner', ['a', 'PAD', 'b'], np.array([0, 1, 2]))


# TODO: test gradcam_text_spans with a small example