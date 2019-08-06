# -*- coding: utf-8 -*-
"""eli5.nn package tests for Grad-CAM."""

import pytest
import numpy as np

from eli5.nn.gradcam import (
    gradcam_heatmap,
)


epsilon = 1e-07


def test_gradcam_zeros():
    shape = (1, 2, 2, 3)
    activations = np.ones(shape) # three 2x2 maps
    grads = np.zeros(shape) # grad for each map
    heatmap = gradcam_heatmap(grads, activations)
    # all zeroes
    assert np.count_nonzero(heatmap) == 0


def test_gradcam_ones():
    shape = (1, 1, 1, 2)
    activations = np.ones(shape)
    grads = np.ones(shape)
    heatmap = gradcam_heatmap(grads, activations)
    expected = np.ones((1, 1))*2 # 2 because we *add* each map
    # all within eps distance
    assert np.isclose(heatmap, expected, rtol=epsilon)


# TODO: test compute_weights with different shapes


# TODO: test get_localization_map with different shapes


# TODO: test relu and counterfactual with simple examples


# TODO: test relu and counterfactual validation


# TODO: test target validation


# def test_get_target_prediction_invalid(simple_seq_image):
#     # only list of targets is currently supported
#     with pytest.raises(TypeError):
#         _get_target_prediction('somestring', simple_seq_image)
#     # only one target prediction is currently supported
#     with pytest.raises(ValueError):
#         _get_target_prediction([1, 2], simple_seq_image)

#     # these are dispatched to _validate_target
#     # only an integer index target is currently supported
#     with pytest.raises(TypeError):
#         _get_target_prediction(['someotherstring'], simple_seq_image)
#     # target index must correctly reference one of the nodes in the final layer
#     with pytest.raises(ValueError):
#         _get_target_prediction([20], simple_seq_image)


# TODO: test_autoget_target_prediction with multiple maximum values, etc
