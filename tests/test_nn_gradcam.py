# -*- coding: utf-8 -*-
"""eli5.nn package tests for Grad-CAM."""

import pytest
import numpy as np
PIL = pytest.importorskip('PIL')

from eli5.nn.gradcam import (
    gradcam_heatmap,
    _validate_targets,
    _validate_classification_target,
    _validate_heatmap,
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


# these might change as the API gets updated
def test_validate_targets():
    with pytest.raises(TypeError):
        # only a list of targets is currently supported
        _validate_targets(1)

    with pytest.raises(ValueError):
        # only one target prediction is currently supported
        _validate_targets([1, 2])

    # only an integer index target is currently supported
    with pytest.raises(TypeError):
        _validate_targets(['somestring'])


# target index must correctly reference one of the nodes in the final layer
def test_validate_classification_target():
    with pytest.raises(ValueError):
        # one over
        _validate_classification_target(2, (1, 2,))
    with pytest.raises(ValueError):
        # one less
        _validate_classification_target(-1, (1, 1,))


def test_validate_heatmap():
    with pytest.raises(TypeError):
        # heatmap must be a numpy array, not a Pillow image
        _validate_heatmap(PIL.Image.new('L', (2, 2,)))
    with pytest.raises(TypeError):
        # heatmap must not be a Python list
        _validate_heatmap([2, 3])