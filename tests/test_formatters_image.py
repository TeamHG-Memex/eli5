# -*- coding: utf-8 -*-

import pytest

PIL = pytest.importorskip('PIL')
matplotlib = pytest.importorskip('matplotlib')

import numpy as np

from eli5.base import Explanation, TargetExplanation
from eli5.formatters.image import (
    format_as_image,
    heatmap_to_image,
    expand_heatmap,
    _validate_heatmap,
    _update_alpha,
    _cap_alpha,
    _overlay_heatmap,
)
from .utils_image import assert_pixel_by_pixel_equal


# 'png' format is required for RGBA data
@pytest.fixture(scope='module')
def boxl():
    return PIL.Image.open('tests/images/box_5x5_l.png')


@pytest.fixture(scope='module')
def boxrgb():
    return PIL.Image.open('tests/images/box_5x5_rgb.png')


@pytest.fixture(scope='module')
def boxrgba():
    return PIL.Image.open('tests/images/box_5x5_rgba.png')


# this is the original catdog image in 'jpg' format with RGB data
@pytest.fixture(scope='module')
def catdog():
    return PIL.Image.open('tests/images/cat_dog.jpg')


@pytest.fixture(scope='module')
def catdog_rgba(catdog):
    return catdog.convert('RGBA')


def test_validate_heatmap(boxl):
    # wrong type
    with pytest.raises(TypeError):
        _validate_heatmap(boxl)
    # out of lower bound
    with pytest.raises(ValueError):
        _validate_heatmap(np.array([-0.001]))
    # out of upper bound
    with pytest.raises(ValueError):
        _validate_heatmap(np.array([1.001]))


@pytest.mark.parametrize('heatmap', [
    (np.zeros((5, 5))),
])
def test_heatmap_to_image_grayscale(heatmap, boxl):
    gray_heatmap = heatmap_to_image(heatmap)
    assert heatmap.shape == (gray_heatmap.width, gray_heatmap.height)
    assert_pixel_by_pixel_equal(gray_heatmap, boxl)


@pytest.mark.parametrize('heatmap', [
    (np.zeros((5, 5, 3))),
])
def test_heatmap_to_image_rgba(heatmap, boxrgb):
    rgba_heatmap = heatmap_to_image(heatmap)
    assert heatmap.shape[:2] == (rgba_heatmap.width, rgba_heatmap.height)
    assert_pixel_by_pixel_equal(rgba_heatmap, boxrgb)


@pytest.mark.parametrize('heatmap', [
    (np.zeros((5, 5, 4))),
])
def test_heatmap_to_image_rgba(heatmap, boxrgba):
    rgba_heatmap = heatmap_to_image(heatmap)
    assert heatmap.shape[:2] == (rgba_heatmap.width, rgba_heatmap.height)
    assert_pixel_by_pixel_equal(rgba_heatmap, boxrgba)


def test_heatmap_to_image_invalid():
    # heatmap must have rank 2 or rank 3
    with pytest.raises(ValueError):
        heatmap_to_image(np.zeros((1,)))
    # coloured heatmap must have 4 or 3 channels
    with pytest.raises(ValueError):
        heatmap_to_image(np.zeros((1, 1, 10)))


@pytest.mark.parametrize('heatmap, colormap', [
    (np.ones((1, 1)), matplotlib.cm.binary),
])
def test_colorize(heatmap, colormap):
    colorized = colormap(heatmap)
    # check rank
    assert len(colorized.shape) == 3
    # check that in interval [0, 1]
    assert colorized.max() <= 1.0
    assert 0.0 <= colorized.min()


@pytest.mark.parametrize('old_arr, alpha_start_arr, new_arr', [
    (np.ones((2, 2, 4)), None, np.ones((2, 2, 4))),
    (np.zeros((1, 1, 4)), np.ones((1, 1)), np.array([[[0, 0, 0, 1]]])),
])
def test_update_alpha(old_arr, alpha_start_arr, new_arr):
    _update_alpha(old_arr, starting_array=alpha_start_arr) # this operation is in-place
    assert np.array_equal(old_arr, new_arr)


@pytest.mark.parametrize('alpha_arr, alpha_limit, new_alpha_arr', [
    (np.zeros((4, 3)), 0, np.zeros((4, 3))),
    (np.array([[0.5, 0.49], [0.51, 0.5]]), 0.5, np.array([[0.5, 0.49], [0.5, 0.5]])),
])
def test_cap_alpha(alpha_arr, alpha_limit, new_alpha_arr):
    capped = _cap_alpha(alpha_arr, alpha_limit)
    assert np.array_equal(capped, new_alpha_arr)


def test_cap_alpha_invalid():
    alpha = np.zeros((1, 1))
    # alpha must be a float or int
    with pytest.raises(TypeError):
        _cap_alpha(alpha, '0.5')
    # alpha must be between 0 and 1
    with pytest.raises(ValueError):
        _cap_alpha(alpha, 1.1)
    with pytest.raises(ValueError):
        _cap_alpha(alpha, -0.1)


@pytest.mark.parametrize('heatmap', [
    (np.zeros((3, 3))),
    (np.zeros((10, 10, 4))), # would need downsizing
])
def test_expand_heatmap(boxrgb, heatmap):
    expanded = expand_heatmap(heatmap, boxrgb, PIL.Image.BOX)
    assert (expanded.width, expanded.height) == (boxrgb.width, boxrgb.height)


def test_expand_heatmap_invalid():
    # image is wrong type
    heatmap = np.zeros((1, 1))
    image = np.ones((2, 2))
    with pytest.raises(TypeError):
        expand_heatmap(heatmap, image, PIL.Image.BOX)


def test_overlay_heatmap(boxrgba):
    overlay = _overlay_heatmap(boxrgba, boxrgba)
    assert_pixel_by_pixel_equal(overlay, boxrgba)


@pytest.fixture(scope='module')
def mock_expl(catdog_rgba):
    return Explanation('mock estimator', 
        image=catdog_rgba, 
        targets=[TargetExplanation(-1, 
            heatmap=np.zeros((7, 7))
    )])


@pytest.fixture(scope='module')
def mock_expl_noheatmap(catdog_rgba):
    return Explanation('mock estimator', 
        image=catdog_rgba, 
    )


@pytest.fixture(scope='module')
def mock_expl_imgarr():
    return Explanation('mock estimator',
        image=np.zeros((2, 2, 4)),
    )


@pytest.fixture(scope='module')
def mock_expl_imgmode(boxl):
    return Explanation('mock estimator',
        image=boxl, # mode 'L'
    )


def test_format_as_image_notransparency(catdog_rgba, mock_expl):
    # heatmap with full transparency
    overlay = format_as_image(mock_expl, alpha_limit=0.0)
    assert_pixel_by_pixel_equal(overlay, catdog_rgba)


def test_format_as_image_noheatmap(catdog_rgba, mock_expl_noheatmap):
    # no heatmap
    overlay = format_as_image(mock_expl_noheatmap)
    assert_pixel_by_pixel_equal(overlay, catdog_rgba)


def test_format_as_image_invalid_expl(mock_expl_imgarr, mock_expl_imgmode):
    # invalid type
    with pytest.raises(TypeError):
        format_as_image(mock_expl_imgarr)
    # invalid image mode
    with pytest.raises(ValueError):
        format_as_image(mock_expl_imgmode)