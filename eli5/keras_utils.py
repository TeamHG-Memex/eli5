# -*- coding: utf-8 -*-

# consider moving this to a 'keras/' directory under the name of 'utils.py'
import numpy as np
from keras.preprocessing import image


def load_image(img, input_shape=None):
    """
    Returns a single image as an array for an estimator's input
    img: one of: path to a single image file, PIL Image object, numpy array
    estimator: model instance, for resizing the image to the required input dimensions
    """
    # TODO: Take in PIL image object, or an array can also be multiple images.
    # "pipeline": path str -> PIL image -> numpy array
    im = image.load_img(img, target_size=input_shape)
    x = image.img_to_array(im)

    # we need to insert an axis at the 0th position to indicate the batch size
    # this is required by the keras predict() function
    x = np.expand_dims(x, axis=0)
    return x