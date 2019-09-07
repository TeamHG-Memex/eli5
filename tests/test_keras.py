# -*- coding: utf-8 -*-
"""Keras unit tests for helpers."""

import pytest

keras = pytest.importorskip('keras')
PIL = pytest.importorskip('PIL')

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import (
    Dense,
    Activation,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    GlobalMaxPooling2D,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    Embedding,
    RNN,
    GRU,
    LSTM,
    Conv1D,
    MaxPooling1D,
    AveragePooling1D,
)
from keras.backend import epsilon
import numpy as np

import eli5
from eli5.keras.explain_prediction import (
    _validate_model,
    _validate_doc,
    _get_layer,
    _autoget_layer_image,
    _autoget_layer_text,
)
from eli5.keras.gradcam import (
    _autoget_target_prediction,
    _calc_gradient,
)


# We need to put this layer in a fixture object AND access it in a parametrization
conv_layer = Conv2D(10, (3, 3))

@pytest.fixture(scope='module')
def simple_seq_image():
    """A simple sequential model for images."""
    model = Sequential([
        Activation('linear', input_shape=(32, 32, 1)), # index 0, input
        conv_layer,                                    # index 1, conv
        Conv2D(20, (3, 3)),                            # index 2, conv2
        GlobalAveragePooling2D(),                      # index 3, gap
        # output shape is (None, 20)
    ])
    enumerate_layers(model)
    print('Summary of model:')
    model.summary()
    return model


@pytest.fixture(scope='module')
def dummy_image():
    image = PIL.Image.new('RGBA', (32, 32))
    print('Dummy image:', image)
    return image


def enumerate_layers(model):
    """Rename ``model`` layers to have name "layerI"
    where "I" is the index, starting from zero,
    going forwards from the input."""
    for i, layer in enumerate(model.layers):
        layer.name = 'layer%d' % i


# layer is the argument to _get_layer
# expected_layer is a unique layer name as a string
@pytest.mark.parametrize('layer, expected_layer', [
    (-3, 'layer1'), # index backwards
    ('layer0', 'layer0'), # name
    (conv_layer, 'layer1'), # instance
])
def test_get_layer(simple_seq_image, layer, expected_layer):
    """Test different ways to specify activation layer."""
    assert _get_layer(simple_seq_image, layer) == simple_seq_image.get_layer(name=expected_layer)


def test_get_layer_invalid(simple_seq_image):
    # invalid layer type
    with pytest.raises(TypeError):
        _get_layer(simple_seq_image, 2.5)
    # note that for invalid layer index or name the underlying 
    # keras get_layer() method raises the appropriate exceptions


@pytest.mark.parametrize('model, expected_layer_idx', [
    (Sequential([Conv2D(1, 1, input_shape=(2, 2, 1,)), AveragePooling2D(1),
        GlobalAveragePooling2D(), ]),
        1),  # match (layer rank backwards)
])
def test_autoget_layer_image(model, expected_layer_idx):
    l = _autoget_layer_image(model)
    assert l is model.get_layer(index=expected_layer_idx)


def test_autoget_layer_image_no_match():
    model = Sequential([Dense(1, input_shape=(2, 3,)), Dense(1), ])
    with pytest.raises(ValueError):
        _autoget_layer_image(model)


@pytest.mark.parametrize('model, expected_layer_idx', [
    (Sequential([Embedding(5, 2), LSTM(1, return_sequences=True), MaxPooling1D(1), ]),
        1),  # text layer
    (Sequential([Embedding(5, 2), MaxPooling1D(1), AveragePooling1D(1), Dense(1), ]),
        2),  # 1D layer backwards
    (Sequential([Embedding(5, 2), Dense(1), ]),
        0),  # embedding
])
def test_autoget_layer_text(model, expected_layer_idx):
    l = _autoget_layer_text(model)
    assert l is model.get_layer(index=expected_layer_idx)


def test_autoget_layer_text_no_match():
    model = Sequential([Dense(1, input_shape=(1,)), Dense(1), ])
    with pytest.raises(ValueError):
        _autoget_layer_text(model)


def test_validate_model_invalid():
    with pytest.raises(ValueError):
        # empty model
        _validate_model(Sequential())


def test_validate_doc():
    with pytest.raises(TypeError):
        _validate_doc(10)
    with pytest.raises(ValueError):
        # batch has more than one sample
        _validate_doc(np.zeros((3, 2, 2, 1)))


def test_explain_prediction_attributes(simple_seq_image, dummy_image):
    expl = eli5.explain_prediction(simple_seq_image, np.zeros((1, 32, 32, 1)))
    assert expl.layer is not None
    assert expl.targets[0].score is not None
    assert expl.targets[0].proba is None


@pytest.mark.parametrize('model, doc', [
    (Sequential(), np.zeros((0,))),  # bad input
    (Sequential(), np.zeros((1, 2, 2, 3),)),  # bad model
])
def test_explain_prediction_not_supported(model, doc):
    res = eli5.explain_prediction(model, doc)
    assert 'supported' in res.error


@pytest.fixture(scope='module')
def differentiable_model():
    inpt = Input(shape=(1,))
    op = Lambda(lambda x: x)(inpt)  # identity function
    model = Model(inpt, op)
    model.summary()
    return model


@pytest.fixture(scope='module')
def nondifferentiable_model():
    inpt = Input(shape=(1,))
    op = Lambda(lambda x: K.constant(0) if x == 0
        else K.constant(1))(inpt) # piecewise function
    model = Model(inpt, op)
    model.summary()
    return model


def test_calc_gradient(differentiable_model):
    _calc_gradient(differentiable_model.output,
        [differentiable_model.input])


def test_calc_gradient_nondifferentiable(nondifferentiable_model):
    with pytest.raises(ValueError):
        grads = _calc_gradient(nondifferentiable_model.output,
            [nondifferentiable_model.input])



# TODO: test_autoget_target_prediction with multiple maximum values, etc


def test_import():
    # test that package imports without errors
    import eli5.keras