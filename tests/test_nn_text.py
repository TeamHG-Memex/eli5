# -*- coding: utf-8 -*-
"""eli5.nn package tests for text helpers."""

import pytest
import numpy as np

from eli5.nn.text import (
    gradcam_spans,
    resize_1d,
    _build_spans,
    _construct_document,
    _find_padding,
    _find_padding_values,
    _find_padding_tokens,
    _trim_padding,
    _validate_tokens,
)
from eli5.base import (
    WeightedSpans,
    DocWeightedSpans,
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


def test_find_padding_invalid():
    # invalid combination
    with pytest.raises(TypeError):
        # pad token and doc
        _find_padding(pad_token='<PAD>', doc=[0, 2, 1], tokens=None)


def test_find_padding_values():
    indices = _find_padding_values(0, np.array([[0, 0, 1, 2]]))
    np.array_equal(indices, np.array([0, 1]))

    with pytest.raises(TypeError):
        _find_padding_values('<PAD>', np.ndarray([0]))


def test_find_padding_tokens():
    indices = _find_padding_tokens('<PAD>', ['the', 'test', '<PAD>', '<PAD>'])
    np.array_equal(indices, np.array([2, 3]))

    with pytest.raises(TypeError):
        _find_padding_tokens(0, ['<PAD>'])


@pytest.mark.parametrize('pad_indices, tokens, heatmap, expected_tokens, expected_heatmap', [
    ([0, 1], ['PAD', 'PAD', 'hi', 'there'], np.array([0.2, 0.1, 2, 3]),
        ['hi', 'there'], np.array([2, 3])),
    ([2, 3], ['hi', 'there', 'PAD', 'PAD'], np.array([2, 3, 0.1, 0.2]),
        ['hi', 'there'], np.array([2, 3])),
])
def test_trim_padding(pad_indices, tokens, heatmap, expected_tokens, expected_heatmap):
    tokens, heatmap = _trim_padding(pad_indices, tokens, heatmap)
    assert np.array_equal(tokens, expected_tokens)
    assert np.array_equal(heatmap, expected_heatmap)


def test_trim_padding_invalid():
    tokens = ['a']
    heatmap = [1]
    pad_indices = []
    tokens_trimmed, heatmap_trimmed = _trim_padding(pad_indices, tokens, heatmap)
    assert np.array_equal(tokens, tokens_trimmed)
    assert np.array_equal(heatmap, heatmap_trimmed)


def test_gradcam_spans():
    heatmap, tokens, doc = np.array([2.0]), ['a'], np.array([[2]])
    res_tokens, res_heatmap, res_weighted_spans = gradcam_spans(heatmap, tokens, doc)
    assert np.array_equal(heatmap, res_heatmap)
    assert np.array_equal(tokens, res_tokens)
    assert res_weighted_spans == WeightedSpans([DocWeightedSpans(
                                                'a',
                                                spans=[('a', [(0, 1)], 2.0)]
                                                )])


def test_validate_tokens():
    _validate_tokens(np.zeros((1, 3)), ['a', 'b', 'c'])
    _validate_tokens(np.zeros((2, 2)), [['a', 'b'], ['c', 'd']])


def test_validate_tokens_invalid():
    with pytest.raises(TypeError):
        # should be in a list
        _validate_tokens(np.zeros((1, 1)), 'a')
    with pytest.raises(ValueError):
        # empty list
        _validate_tokens(np.zeros((1, 1)), [])
    with pytest.raises(ValueError):
        # single list but multiple samples in batch
        _validate_tokens(np.zeros((3, 2)), ['a', 'b'])

    # list doesn't contain strings
    with pytest.raises(TypeError):
        _validate_tokens(np.zeros((1, 1)), [0])
    with pytest.raises(TypeError):
        _validate_tokens(np.zeros((1, 1)), [[0]])

    with pytest.raises(ValueError):
        # not enough samples in batched list
        _validate_tokens(np.zeros((3, 1)), np.array([['a'], ['b']]))
    with pytest.raises(ValueError):
        # tokens lengths vary
        _validate_tokens(np.zeros((2, 2)), [['a', 'b'], ['c']])
    with pytest.raises(ValueError):
        # tokens sample lengths do not match
        _validate_tokens(np.zeros((1, 1)), ['a', 'b'])
    with pytest.raises(TypeError):
        # too many axes
        _validate_tokens(np.zeros((1, 1,)), [[['a']]])