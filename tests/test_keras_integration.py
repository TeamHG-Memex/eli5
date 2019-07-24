# -*- coding: utf-8 -*-

"""Test integration of Grad-CAM explanation and image formatter for Keras"""
from __future__ import print_function

import pytest

keras = pytest.importorskip('keras')
PIL = pytest.importorskip('PIL')
matplotlib = pytest.importorskip('matplotlib')
IPython = pytest.importorskip('IPython')

import numpy as np
from PIL import Image
from keras.applications import (
    mobilenet_v2
)

import eli5
from eli5.base import Explanation
from eli5 import format_as_image
from eli5.formatters.image import (
    expand_heatmap,
    _normalize_heatmap,
)
from .utils_image import assert_pixel_by_pixel_equal


imagenet_cat_idx = 282


@pytest.fixture(scope='module')
def keras_clf():
    # This is a small classifier (~14 MB, ~3.5 million weights).
    # On first run weights are downloaded automatically and cached.
    # See https://keras.io/applications/
    clf = mobilenet_v2.MobileNetV2(alpha=1.0, include_top=True, weights='imagenet', classes=1000)
    print('Summary of classifier:')
    clf.summary()
    return  clf


@pytest.fixture(scope='module')
def cat_dog_image():
    # Note that 'jpg' images can have RGB data
    # which is fine in the case of this model (requires three channels)
    img_path = 'tests/images/cat_dog.jpg'
    im = keras.preprocessing.image.load_img(img_path, target_size=(224, 224), color_mode='rgba')
    doc_im = im.convert(mode='RGB') # TODO: might be good idea to take any mode image, not just rgba
    doc = keras.preprocessing.image.img_to_array(doc_im)
    doc = np.expand_dims(doc, axis=0)
    mobilenet_v2.preprocess_input(doc) # because we our classifier is mobilenet_v2
    return doc, im


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
    heatmap = expl.targets[0].heatmap

    # fit heatmap over image
    heatmap = _normalize_heatmap(heatmap)
    heatmap = expand_heatmap(heatmap, image, Image.LANCZOS)
    heatmap = np.array(heatmap)

    # get a slice of the area
    x1, x2, y1, y2 = area
    crop = heatmap[y1:y2, x1:x2] # row-first ordering
    # TODO: instead of hard-coding the height and width offsets
    # it might be a better idea to use percentages
    # this makes the tests independent of any resizing done on the image
    # and the heatmap doesn't have to be resized
    # however, it might be harder for the user to determine percentages

    # check intensity
    total_intensity = np.sum(heatmap)
    crop_intensity = np.sum(crop)
    p = total_intensity / 100 # -> 1% of total_intensity
    crop_p = crop_intensity / p # -> intensity %
    n = 40 # at least n% (need to experiment with this number)
    assert n < crop_p

    # Alternatively, check that the intensity over area 
    # is greater than all other intensity:
    # remaining_intensity = total_intensity - intensity
    # assert remaining_intensity < total_intensity


# area = (x1, x2, y1, y2)
@pytest.mark.parametrize('area, targets', [
    ((54, 170, 2, 100), None), # focus on the dog (pick top prediction)
    ((44, 180, 130, 212), [imagenet_cat_idx]), # focus on the cat (supply prediction)
])
def test_image_classification(keras_clf, cat_dog_image, area, targets):
    # check explanation
    res = eli5.explain_prediction(keras_clf, cat_dog_image[0], 
        image=cat_dog_image[1], targets=targets)
    assert_attention_over_area(res, area)
    
    # check formatting
    overlay = format_as_image(res)
    # import matplotlib.pyplot as plt; plt.imshow(overlay); plt.show()
    assert_good_external_format(res, overlay)

    # check show function
    show_overlay = eli5.show_prediction(keras_clf, cat_dog_image[0], 
        image=cat_dog_image[1], targets=targets)
    assert_pixel_by_pixel_equal(overlay, show_overlay)


@pytest.fixture(scope="function")
def show_nodeps(request):
    # register tear down
    old_format_as_image = eli5.ipython.format_as_image
    def fin():
        # tear down
        eli5.ipython.format_as_image = old_format_as_image
    request.addfinalizer(fin)

    # set up
    eli5.ipython.format_as_image = ImportError('mock test')

    # object return
    yield eli5.show_prediction


def test_show_prediction_nodeps(show_nodeps, keras_clf, cat_dog_image):
    with pytest.warns(UserWarning):
        expl = show_nodeps(keras_clf, cat_dog_image[0], image=cat_dog_image[1])
    assert isinstance(expl, Explanation)


# TODO: test no relu and counterfactual explanations