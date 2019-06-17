# -*- coding: utf-8 -*-

"""Test integration of Grad-CAM explanation and image formatter for Keras"""

import pytest

keras = pytest.importorskip('keras')
PIL = pytest.importorskip('PIL')
IPython = pytest.importorskip('IPython')

import numpy as np
from PIL import Image
from keras.applications import (
    mobilenet_v2
)

import eli5
from eli5 import format_as_image
from eli5.keras import image_from_path
from eli5.formatters.image import (
    heatmap_to_grayscale,
    resize_over
)
from .utils_image import assert_pixel_by_pixel_equal


# TODO: time these tests

imagenet_cat_idx = 282


@pytest.fixture(scope='module')
def keras_clf():
    # TODO: load weights from a file
    return mobilenet_v2.MobileNetV2(alpha=1.0, include_top=True, weights='imagenet', classes=1000)


@pytest.fixture(scope='module')
def cat_dog_image():
    # TODO: consider a .png example
    doc = image_from_path('tests/images/cat_dog.jpg', image_shape=(224, 224))
    doc = mobilenet_v2.preprocess_input(doc) # FIXME: this preprocessing is hardcoded for mobilenet_v2
    return doc


def assert_good_external_format(expl, overlay):
    """
    Check properties of the formatted heatmap over the original image,
    using external properties of the image,
    such as dimensions, mode, type.
    """
    original = expl.image
    # check external properties
    assert isinstance(overlay, Image.Image)
    assert overlay.width == original.width
    assert overlay.height == original.height
    assert overlay.mode == 'RGBA'


def assert_attention_over_area(expl, area):
    """
    Check that the explanation 'expl' lights up the most over 'area', 
    a tuple of (x1, x2, y1, y2), starting and ending points of the bounding rectangle
    in the original image.
    We make two assumptions in this test:
    1. The model can classify the example image correctly.
    2. The area specified by the tester over the example image covers the predicted class correctly.
    """
    image = expl.image
    heatmap = expl.heatmap
    # fit heatmap over image
    # FIXME: this might be too circular? Need to test image formatter first?
    heatmap = heatmap_to_grayscale(heatmap)
    heatmap = resize_over(heatmap, image, interpolation=Image.LANCZOS)
    heatmap = np.array(heatmap)
    x1, x2, y1, y2 = area
    crop = heatmap[y1:y2, x1:x2] # row-first ordering

    # TODO: show image, heatmap, and overlay when test fails
    # https://stackoverflow.com/questions/13364868/in-pytest-how-can-i-figure-out-if-a-test-failed-from-request
    # https://docs.pytest.org/en/latest/example/simple.html#making-test-result-information-available-in-fixtures
    # https://stackoverflow.com/questions/35703122/how-to-detect-when-pytest-test-case-failed/36219273
    # Current manual solution:
    # import matplotlib.pyplot as plt; plt.imshow(im); plt.show()

    total_intensity = np.sum(heatmap)
    crop_intensity = np.sum(crop)
    p = total_intensity / 100 # -> 1% of total_intensity
    intensity = crop_intensity / p # -> intensity %
    assert 50 < intensity # at least 50% (need to experiment with this number)
    # Alternative:
    # remaining_intensity = total_intensity - intensity
    # assert remaining_intensity < total_intensity


# area = (x1, x2, y1, y2)
# TODO: instead of hard-coding height and width pixels, be able to take percentages
@pytest.mark.parametrize('area, targets', [
    ((54, 170, 2, 100), None), # focus on the dog (pick top prediction)
    ((44, 180, 130, 212), [imagenet_cat_idx]), # focus on the cat (supply prediction)
])
def test_image_classification(keras_clf, cat_dog_image, area, targets):
    # check explanation
    res = eli5.explain_prediction(keras_clf, cat_dog_image, targets=targets)
    assert_attention_over_area(res, area)
    
    # check formatting
    overlay = format_as_image(res)
    
    assert_good_external_format(res, overlay)

    # check show function
    show_overlay = eli5.show_prediction(keras_clf, cat_dog_image, targets=targets)
    assert_pixel_by_pixel_equal(overlay, show_overlay)