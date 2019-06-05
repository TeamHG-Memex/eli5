# -*- coding: utf-8 -*-

import pytest
import numpy as np
import PIL
import matplotlib.pyplot as plt
from keras.applications import (
    mobilenet_v2
)

import eli5
from eli5.keras_utils import load_image


imagenet_cat_idx = 282


@pytest.fixture(scope='module')
def classifier():
    # TODO: load weights from a file
    return mobilenet_v2.MobileNetV2(alpha=1.0, include_top=True, weights='imagenet', classes=1000)


@pytest.fixture(scope='module')
def image():
    doc = load_image('images/cat_dog.jpg', (224, 224))
    doc = mobilenet_v2.preprocess_input(doc) # FIXME: this preprocessing is hardcoded for mobilenet_v2
    return doc


def assert_attention_over_area(expl, area):
    """
    Explanation 'expl' over 'area', 
    a tuple of (x1, x2, y1, y2), starting and ending points of the bounding rectangle.
    """
    image = expl.image
    heatmap = expl.heatmap
    # fit heatmap over image
    heatmap = heatmap.resize((image.width, image.height), resample=PIL.Image.LANCZOS)
    heatmap = np.array(heatmap)
    x1, x2, y1, y2 = area
    focus = heatmap[y1:y2, x1:x2] # row-first ordering

    # TODO: show formatted image / heatmap image if test fails
    plt.imshow(image); plt.show()
    plt.imshow(heatmap); plt.show()
    plt.imshow(focus); plt.show()

    total_intensity = np.sum(heatmap)
    area_intensity = np.sum(focus)
    p = total_intensity / 100 # -> 1% of total_intensity
    intensity = area_intensity / p # -> intensity %
    # assert 70 < intensity # at least 70% (experiment with this number) # FIXME: fails for cat example
    remaining_intensity = total_intensity - intensity
    assert remaining_intensity < total_intensity


# TODO: consider a .png example
# TODO: time these tests
# area = (x1, x2, y1, y2)
@pytest.mark.parametrize('area, targets', [
    ((54, 170, 2, 100), None), # focus on the dog (pick top prediction)
    ((44, 180, 130, 212), [imagenet_cat_idx]) # focus on the cat (pass prediction)
])
def test_gradcam_image_classification(classifier, image, area, targets):
    res = eli5.explain_prediction(classifier, image, targets=targets)
    assert_attention_over_area(res, area)