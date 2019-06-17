# -*- coding: utf-8 -*-

import pytest

PIL = pytest.importorskip('PIL')
matplotlib = pytest.importorskip('matplotlib')

import numpy as np

from eli5.formatters.image import (
    heatmap_to_grayscale,
    heatmap_to_rgba,
)
from .utils_image import assert_pixel_by_pixel_equal

# TODO: test format_as_image with alpha_limit set to 0 -> should show no heatmap over the image.


# heatmap is a rank 2 [0, 1] numpy array
# grayscale image is a path to expected image
@pytest.mark.parametrize('heatmap, expected_im', [
    (np.zeros((5, 5)), PIL.Image.open('tests/images/black_5x5.jpg')),
])
def test_heatmap_to_grayscale(heatmap, expected_im):
    gray_heatmap = heatmap_to_grayscale(heatmap)
    assert heatmap.shape == (gray_heatmap.width, gray_heatmap.height)
    assert_pixel_by_pixel_equal(gray_heatmap, expected_im)

# TODO: test validation of values / heatmap


@pytest.mark.parametrize('heatmap, expected_im', [
    (np.zeros((5, 5, 4)), PIL.Image.open('tests/images/black_5x5.png')),
])
def test_heatmap_to_rgba(heatmap, expected_im):
    rgba_heatmap = heatmap_to_rgba(heatmap)
    assert heatmap.shape[:2] == (rgba_heatmap.width, rgba_heatmap.height)
    assert_pixel_by_pixel_equal(rgba_heatmap, expected_im)



# def test_colorize


# def test_update_alpha


# def test_cap_alpha


# def test_resize_over


# def test_convert_image


# def test_format_as_image