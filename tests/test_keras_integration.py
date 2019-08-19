# -*- coding: utf-8 -*-
"""Test integration of Grad-CAM explanation and image formatter for Keras."""
from __future__ import print_function

import pytest

keras = pytest.importorskip('keras')
PIL = pytest.importorskip('PIL')
matplotlib = pytest.importorskip('matplotlib')
IPython = pytest.importorskip('IPython')

import numpy as np
from keras.applications import (
    mobilenet_v2
)
from keras.models import Sequential

import eli5
from eli5.base import Explanation
from eli5 import format_as_image
from .utils_image import assert_pixel_by_pixel_equal
from .utils_gradcam import (
    assert_good_external_format,
    assert_attention_over_area,
)


imagenet_cat_idx = 282


@pytest.fixture(scope='module')
def keras_clf():
    # This is a small classifier (~14 MB, ~3.5 million weights).
    # On first run weights are downloaded automatically and cached.
    # See https://keras.io/applications/
    clf = mobilenet_v2.MobileNetV2(alpha=1.0, include_top=True, weights='imagenet', classes=1000)
    print('Summary of classifier:')
    clf.summary()
    return clf


@pytest.fixture(scope='module')
def cat_dog_image():
    # Note that 'jpg' images can have RGB data
    # which is fine in the case of this model (requires three channels)
    img_path = 'tests/images/cat_dog.jpg'
    im = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    doc = keras.preprocessing.image.img_to_array(im)
    doc = np.expand_dims(doc, axis=0)  # add batch size
    mobilenet_v2.preprocess_input(doc)  # because our classifier is mobilenet_v2
    # re-load from array because we did some preprocessing
    im = keras.preprocessing.image.array_to_img(doc[0])
    return doc, im


# area = (x1, x2, y1, y2)
dog_area = (54, 170, 2, 100)
cat_area = (44, 180, 130, 212)
all_area = (0, 224, 0, 224)


@pytest.mark.parametrize('targets, area', [
    (None, dog_area),  # focus on the dog (pick top prediction)
    ([imagenet_cat_idx], cat_area),  # focus on the cat (supply prediction)
])
def test_image_classification(keras_clf, cat_dog_image, targets, area):
    doc, image = cat_dog_image
    # check explanation
    res = eli5.explain_prediction(keras_clf, doc, image=image, targets=targets)
    assert_attention_over_area(res, area)

    # check formatting
    # explicitly normalize to test
    res.image = res.image.convert('RGBA')
    overlay = format_as_image(res)
    # import matplotlib.pyplot as plt; plt.imshow(overlay); plt.show()
    assert_good_external_format(res, overlay)


# check explanation modifiers 'relu' and 'counterfactual'
@pytest.mark.parametrize('targets, relu, counterfactual, area, limit, invert', [
    (None, True, False, all_area, 90, False),  # everything is highlighted
    (None, False, True, dog_area, 60, True),  # dog is not highlighted
    ([imagenet_cat_idx], False, True, cat_area, 80, True),  # cat is not highlighted
])
def test_explain_relu_counterfactual(keras_clf, cat_dog_image, targets,
                                     relu, counterfactual,
                                     area, limit, invert):
    doc, image = cat_dog_image
    res = eli5.explain_prediction(keras_clf, doc, image=image, targets=targets,
                                  relu=relu, counterfactual=counterfactual)
    assert_attention_over_area(res, area, invert=invert)


def test_show_prediction(keras_clf, cat_dog_image):
    doc, image = cat_dog_image
    # explain + format
    res = eli5.explain_prediction(keras_clf, doc, image=image)
    overlay = format_as_image(res)

    # show
    # with image auto-conversion to test
    show_overlay = eli5.show_prediction(keras_clf, doc)
    assert_pixel_by_pixel_equal(overlay, show_overlay)


# explain dense layers
def test_explain_1d_layer_image(keras_clf, cat_dog_image):
    doc, image = cat_dog_image
    eli5.explain_prediction(keras_clf, doc, layer=-1)


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
    doc, image = cat_dog_image
    with pytest.warns(UserWarning) as rec:
        expl = show_nodeps(keras_clf, doc)
    assert 'dependencies' in str(rec[-1].message)
    assert isinstance(expl, Explanation)