# -*- coding: utf-8 -*-

import pytest

PIL = pytest.importorskip('PIL')
matplotlib = pytest.importorskip('matplotlib')

import numpy as np

from eli5.formatters.image import (
    heatmap_to_grayscale,
    heatmap_to_rgba,
    update_alpha,
    cap_alpha,
)
from .utils_image import assert_pixel_by_pixel_equal

# TODO: test format_as_image with alpha_limit set to 0 -> should show no heatmap over the image.


@pytest.mark.parametrize('heatmap, expected_im', [
    (np.zeros((5, 5)), PIL.Image.open('tests/images/black_5x5.jpg')),
])
def test_heatmap_to_grayscale(heatmap, expected_im):
    gray_heatmap = heatmap_to_grayscale(heatmap)
    assert heatmap.shape == (gray_heatmap.width, gray_heatmap.height)
    assert_pixel_by_pixel_equal(gray_heatmap, expected_im)

# TODO: test validation of heatmap argument


@pytest.mark.parametrize('heatmap, expected_im', [
    (np.zeros((5, 5, 4)), PIL.Image.open('tests/images/black_5x5.png')),
])
def test_heatmap_to_rgba(heatmap, expected_im):
    rgba_heatmap = heatmap_to_rgba(heatmap)
    assert heatmap.shape[:2] == (rgba_heatmap.width, rgba_heatmap.height)
    assert_pixel_by_pixel_equal(rgba_heatmap, expected_im)


@pytest.mark.parametrize('old_arr, alpha_start_arr, new_arr', [
    (np.ones((2, 2, 4)), None, np.ones((2, 2, 4))),
    (np.zeros((1, 1, 4)), np.ones((1, 1)), np.array([[[0, 0, 0, 1]]])),
])
def test_update_alpha(old_arr, alpha_start_arr, new_arr):
    update_alpha(old_arr, starting_array=alpha_start_arr) # this operation is in-place
    assert np.array_equal(old_arr, new_arr)


@pytest.mark.parametrize('alpha_arr, alpha_limit, new_alpha_arr', [
    # (np.zeros((3, 3)), None, np.zeros((3, 3))), # this is already covered by test_update_alpha
    (np.zeros((4, 3)), 0, np.zeros((4, 3))),
    (np.array([[0.5, 0.49], [0.51, 0.5]]), 0.5, np.array([[0.5, 0.49], [0.5, 0.5]])),
])
def test_cap_alpha(alpha_arr, alpha_limit, new_alpha_arr):
    capped = cap_alpha(alpha_arr, alpha_limit)
    assert np.array_equal(capped, new_alpha_arr)


def test_cap_alpha_invalid():
    alpha = np.zeros((1, 1))
    with pytest.raises(TypeError):
        cap_alpha(alpha, '0.5')
    # alpha must be between 0 and 1
    with pytest.raises(ValueError):
        cap_alpha(alpha, 1.1)
    with pytest.raises(ValueError):
        cap_alpha(alpha, -0.1)

# TODO: test invalid array shape


# def test_colorize
# TODO: test colorize with a callable

# def test_resize_over


# def test_convert_image


# def test_format_as_image