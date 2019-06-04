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


@pytest.fixture(scope='module')
def classifier():
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

    total_intensity = np.sum(heatmap)
    area_intensity = np.sum(focus)
    p = total_intensity / 100
    intensity = area_intensity / p
    # TODO: show formatted image / heatmap image if test fails
    assert 70 < intensity # at least 70% (experiment with this number)


# TODO: consider a .png example
@pytest.mark.parametrize('area', [
    (((54, 170, 2, 100))), # focus on the dog
])
def test_gradcam_image_classification(classifier, image, area):
    res = eli5.explain_prediction(classifier, image)
    assert_attention_over_area(res, area)