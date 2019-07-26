# -*- coding: utf-8 -*-
"""eli5.nn package tests for Grad-CAM"""

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