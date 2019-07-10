# -*- coding: utf-8 -*-

# image related utilities

import PIL
import numpy as np


def assert_pixel_by_pixel_equal(im1, im2):
    """
    Check that two PIL images are equal
    pixel-by-pixel.
    """
    # see https://stackoverflow.com/questions/1927660/compare-two-images-the-python-linux-way
    # compute pixel-by-pixel difference
    diff = PIL.ImageChops.difference(im1, im2)
    # if no difference, array is all 0's
    diffa = np.array(diff)
    assert np.count_nonzero(diffa) == 0
