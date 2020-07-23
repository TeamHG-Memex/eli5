# -*- coding: utf-8 -*-

"""Grad-CAM integration test related utilities."""

from PIL import Image
import numpy as np

from eli5.formatters.image import (
    heatmap_to_image,
    expand_heatmap,
)


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


def assert_attention_over_area(expl, area, n=40, invert=False):
    """
    Check that the explanation `expl` lights up the most over `area`,
    a tuple of (x1, x2, y1, y2), starting and ending points of the bounding rectangle
    in the original image (x is horizontal, y is vertical).
    We make two assumptions in this test:
    1. The model can classify the example image correctly.
    2. The area specified by the tester over the example image covers the predicted class correctly.

    `n` is percentage of intensity that should be in the area at least.

    `invert` controls whether checking the intensity inside the area (`False`) or outside (`True`).
    """
    image = expl.image
    heatmap = expl.targets[0].heatmap

    # fit heatmap over image
    heatmap = expand_heatmap(heatmap_to_image(heatmap), image, Image.LANCZOS)
    heatmap = np.array(heatmap)

    # get a slice of the area
    x1, x2, y1, y2 = area
    crop = heatmap[y1:y2, x1:x2]  # row-first ordering
    # TODO: Use percentages intead of hard-coded values

    # check intensity
    total_intensity = np.sum(heatmap)
    crop_intensity = np.sum(crop)
    p = total_intensity / 100  # -> 1% of total_intensity
    crop_p = crop_intensity / p  # -> intensity %
    if invert:
        crop_p = 100 - crop_p  # take complement (check intensity outside area)
    assert n < crop_p  # at least n% (need to experiment with this number)

    # Alternatively, check that the intensity over area
    # is greater than all other intensity:
    # remaining_intensity = total_intensity - intensity
    # assert remaining_intensity < total_intensity