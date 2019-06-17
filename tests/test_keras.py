# -*- coding: utf-8 -*-

"""Keras unit tests"""

import pytest

keras = pytest.importorskip('keras')

from keras.models import Sequential
from keras.layers import Activation, Conv2D, GlobalAveragePooling2D
import numpy as np

from eli5.keras import (
    validate_doc,
    get_activation_layer,
    image_from_path,
    get_target_prediction,
)


# We need to put this layer in a fixture object AND access it in a parametrization
gap_layer = GlobalAveragePooling2D()

@pytest.fixture(scope='module')
def simple_seq():
    """A simple sequential model for images."""
    model = Sequential([
        Activation('linear', input_shape=(32, 32, 1)), # index 0, input
        Conv2D(10, (3, 3)),                            # index 1, conv
        Conv2D(20, (3, 3)),                            # index 2, conv2
        gap_layer,                                     # index 3, gap
    ])
    # model.summary() # TODO: print model summary if fail
    # rename layers
    for i, layer in enumerate(model.layers):
        layer.name = 'layer%d' % i
    return model


# layer is an argument to get_activation_layer
# expected_layer is a unique layer name
@pytest.mark.parametrize('layer, expected_layer', [
    (-3, 'layer1'), # index backwards
    ('layer0', 'layer0'),
    (gap_layer, 'layer3'),
    (None, 'layer2'), # first matching layer going back from output layer
])
def test_get_activation_layer(simple_seq, layer, expected_layer):
    """Test different ways to specify activation layer, and automatic activation layer getter"""
    assert get_activation_layer(simple_seq, layer) == simple_seq.get_layer(name=expected_layer)


# note that cases where an invalid layer index or name is passed are 
# handled by the underlying keras get_layer method
def test_get_activation_layer_invalid(simple_seq):
    with pytest.raises(TypeError):
        get_activation_layer(simple_seq, 2.5) # some nonsense


def test_get_activation_layer_unfound(simple_seq):
    with pytest.raises(ValueError):
        get_activation_layer(
            Sequential(), # a model with no layers
            None,
        )
        # this is handled by search_layer_backwards function


# dims = (height, width)
# FIXME: different ordering?
@pytest.mark.parametrize('img_path, dims, expected_shape', [
    ('tests/images/cat_dog.jpg', (30, 50), (1, 30, 50, 3)), # rank 4, RGB
])
def test_image_from_path(img_path, dims, expected_shape):
    im = image_from_path(img_path, image_shape=dims)
    assert im.shape == expected_shape


def test_validate_doc(simple_seq):
    # should raise no errors
    doc = np.zeros((1, 32, 32, 1))
    validate_doc(simple_seq, doc)


def test_validate_doc_multisample(simple_seq):
    # batch has more than one sample
    multisample = np.zeros((3, 32, 32, 1))
    with pytest.raises(ValueError):
        validate_doc(simple_seq, multisample)


def test_validate_doc_wrongdims(simple_seq):
    wrongdims = np.zeros((5, 5))
    with pytest.raises(ValueError):
        validate_doc(simple_seq, wrongdims)

 
def test_get_target_prediction_invalid():
    output = keras.backend.variable(np.zeros((1, 20)))
    with pytest.raises(ValueError):
        get_target_prediction([1, 2], output)
    with pytest.raises(TypeError):
        get_target_prediction('somestring', output)
    # FIXME: both of these exceptions may change or be removed in the future


# TODO: test get_target_prediction() once it is finalized regarding non-classification models

# TODO: test get_target_prediction() with more than one prediction target

# TODO: test invalid argument to targets

# TODO: test invalid prediction ID

# TODO: test grad_cam() with lmap all 0's