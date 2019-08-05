# -*- coding: utf-8 -*-

"""Keras unit tests"""

import pytest

keras = pytest.importorskip('keras')

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import (
    Dense, 
    Activation, 
    Conv2D, 
    GlobalAveragePooling2D, 
    Input, 
    Lambda,
)
from keras.backend import epsilon
import numpy as np

from eli5.keras.explain_prediction import (
    explain_prediction,
    _validate_doc,
    _get_activation_layer,
)
from eli5.keras.gradcam import (
    _get_target_prediction,
    _calc_gradient,
    gradcam,
)


# We need to put this layer in a fixture object AND access it in a parametrization
conv_layer = Conv2D(10, (3, 3))

@pytest.fixture(scope='module')
def simple_seq():
    """A simple sequential model for images."""
    model = Sequential([
        Activation('linear', input_shape=(32, 32, 1)), # index 0, input
        conv_layer,                                    # index 1, conv
        Conv2D(20, (3, 3)),                            # index 2, conv2
        GlobalAveragePooling2D(),                      # index 3, gap
        # output shape is (None, 20)
    ])
    print('Summary of model:')
    model.summary()
    # rename layers
    for i, layer in enumerate(model.layers):
        layer.name = 'layer%d' % i
    return model


# layer is the argument to _get_activation_layer
# expected_layer is a unique layer name
@pytest.mark.parametrize('layer, expected_layer', [
    (-3, 'layer1'), # index backwards
    ('layer0', 'layer0'), # name
    (conv_layer, 'layer1'), # instance
    (None, 'layer2'), # automatic, first matching layer going back from output layer
])
def test_get_activation_layer(simple_seq, layer, expected_layer):
    """Test different ways to specify activation layer, and automatic activation layer getter"""
    assert _get_activation_layer(simple_seq, layer) == simple_seq.get_layer(name=expected_layer)


def test_get_activation_layer_invalid(simple_seq):
    # invalid layer shape
    with pytest.raises(ValueError):
        # GAP has rank 2 shape, need rank 4
        _get_activation_layer(simple_seq, 'layer3')
    # invalid layer type
    with pytest.raises(TypeError):
        _get_activation_layer(simple_seq, 2.5)
    # can not find activation layer automatically
    # this is handled by _search_layer_backwards function
    with pytest.raises(ValueError):
        _get_activation_layer(
            Sequential(), # a model with no layers
            None,
        )
    
    # note that cases where an invalid layer index or name is passed are 
    # handled by the underlying keras get_layer method()


def test_validate_doc(simple_seq):
    # should raise no errors
    _validate_doc(simple_seq, np.zeros((1, 32, 32, 1)))
    # batch has more than one sample
    with pytest.raises(ValueError):
        _validate_doc(simple_seq, np.zeros((3, 32, 32, 1)))
    # type is wrong
    with pytest.raises(TypeError):
        _validate_doc(simple_seq, 10)
    # incorrect dimensions
    with pytest.raises(ValueError):
        _validate_doc(simple_seq, np.zeros((5, 5)))


def test_validate_doc_custom():
    # model with custom (not rank 4) input shape
    model = Sequential([Dense(1, input_shape=(2, 3))])
    # not matching shape
    with pytest.raises(ValueError):
        _validate_doc(model, np.zeros((5, 3)))

 
def test_get_target_prediction_invalid(simple_seq):
    # only list of targets is currently supported
    with pytest.raises(TypeError):
        _get_target_prediction('somestring', simple_seq)
    # only one target prediction is currently supported
    with pytest.raises(ValueError):
        _get_target_prediction([1, 2], simple_seq)

    # these are dispatched to _validate_target
    # only an integer index target is currently supported
    with pytest.raises(TypeError):
        _get_target_prediction(['someotherstring'], simple_seq)
    # target index must correctly reference one of the nodes in the final layer
    with pytest.raises(ValueError):
        _get_target_prediction([20], simple_seq)


def test_explain_prediction_score(simple_seq):
    expl = explain_prediction(simple_seq, np.zeros((1, 32, 32, 1)))
    assert expl.targets[0].score is not None
    assert expl.targets[0].proba is None


@pytest.fixture(scope='module')
def differentiable_model():
    inpt = Input(shape=(1,))
    op = Lambda(lambda x: x)(inpt) # identity function
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


def test_gradcam_zeros():
    activations = np.zeros((2, 2, 3)) # three 2x2 maps
    weights = np.zeros((3,)) # weight for each map
    lmap = gradcam(weights, activations)
    # all zeroes
    assert np.count_nonzero(lmap) == 0


def test_gradcam_ones():
    activations = np.ones((1, 1, 2))
    weights = np.ones((2,))
    lmap = gradcam(weights, activations)
    # all within eps distance to one
    assert np.isclose(lmap, np.ones((1, 1)), rtol=epsilon())