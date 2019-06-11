# -*- coding: utf-8 -*-

import numpy as np # type: ignore
from PIL import Image # type: ignore
import matplotlib.cm # type: ignore


def format_as_image(expl,
    interpolation=Image.LANCZOS,
    colormap=matplotlib.cm.magma,
    alpha_limit=165.75,
    ):
    """ Format :class:`base.Explanation` object as an image.
    
    Parameters
    ----------
    interpolation: int, optional
        Interpolation ID / Pillow filter to use when resizing the image.

        Default is PIL.Image.LANCZOS.

    colormap: object, optional
        Colormap scheme to be applied when converting the heatmap from grayscale to RGB.
        Either a colormap from matplotlib.cm, 
        or a callable that takes a rank 2 array and 
        returns the colored heatmap as a [0, 1] RGBA numpy array.

        Default is matplotlib.cm.magma (blue to red).

    alpha_limit: float, optional
        Maximum alpha (transparency / opacity) value allowed 
        for the alpha channel pixels in the RGBA heatmap image.
        Between 0.0 and 255.0.

        Useful when laying the heatmap over the original image, 
        so that the image can be seen over the heatmap.

        Default is alpha_limit=165.75.

    Returns
    -------
    overlay : object
        PIL image instance of the heatmap blended over the image.
    """
    image = expl.image
    heatmap = expl.heatmap
    
    heatmap = resize_over(heatmap, image, interpolation=interpolation)
    heatmap = np.array(heatmap) # PIL image -> ndarray [0, 255] spatial map
    heatmap_grayscale = heatmap # save the 'pre-colormap' (grayscale) heatmap
    heatmap = colorize(heatmap, colormap=colormap) # apply color map
    # TODO: test colorize with a callable

    # make the alpha intensity correspond to the grayscale heatmap values
    # cap the intensity so that it's not too opaque when near maximum value
    # TODO: more options for controlling alpha, i.e. a callable?
    heatmap = update_alpha(heatmap, starting_array=heatmap_grayscale, alpha_limit=alpha_limit)
    overlay = overlay_heatmap(heatmap, image)
    # TODO: keep types consistent, i.e. in this function only deal with PIL images
    # instead of switching between PIL and numpy arrays
    return overlay


def resize_over(heatmap, image, interpolation):
    """ 
    Resize the `heatmap` image to fit over the original `image`,
    using the specified `interpolation` method.

    See :func:`eli5.format_as_image` for more details on the `interpolation` parameter.
    
    Returns
    -------
    resized_image : object
        A resized PIL image.
    """
    # PIL seems to have a much nicer API for resizing than scipy (scipy.ndimage)
    # Also, scipy seems to have some interpolation problems: 
    # https://github.com/scipy/scipy/issues/8210
    spatial_dimensions = (image.height, image.width)
    heatmap = heatmap.resize(spatial_dimensions, resample=interpolation)
    return heatmap


def colorize(heatmap, colormap):
    """
    Apply `colormap` to a grayscale `heatmap`. 

    See :func:`eli5.format_as_image` for more details on the `colormap` parameter.

    Returns
    -------
    new_heatmap : object
        An RGBA [0, 255] ndarray.
    """
    heatmap = colormap(heatmap) # -> [0, 1] RGBA ndarray
    heatmap = np.uint8(heatmap*255) # re-scale: [0, 1] -> [0, 255] ndarray
    return heatmap


def update_alpha(image_array, starting_array=None, alpha_limit=None):
    """
    Update the alpha channel values of an RGBA ndarray `image_array`,
    optionally creating the alpha channel from `starting_array`
    and setting upper limit for alpha values (opacity) to `alpha_limit`.

    See :func:`eli5.format_as_image` for more details on the `alpha_limit` parameter.

    Returns 
    -------
    new_image : object
        the original `image_array` with the updated alpha channel.
    """
    if isinstance(starting_array, np.ndarray):
        alpha = starting_array
    else:
        # take the alpha channel as is
        alpha = image_array[:,:,3]
    if alpha_limit is not None:
        alpha = np.minimum(alpha, alpha_limit)
    image_array[:,:,3] = alpha
    return image_array
    # TODO: optimisation?


def convert_image(img):
    """ 
    Convert the `img` np.ndarray or PIL.Image.Image instance to an RGBA PIL Image.
    
    Returns
    -------
    pil_image : object
        A PIL image object.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img) # ndarray -> PIL image
    if isinstance(img, Image.Image):
        if img.mode == 'RGB':
            img = img.convert(mode='RGBA') # RGB image -> RGBA image
    return img


def overlay_heatmap(heatmap, image):
    """
    Blend 'heatmap' over 'image'.
    
    Returns
    -------
    overlayed_image : object
        A blended PIL image.
    """
    # normalise to same format
    heatmap = convert_image(heatmap)
    image = convert_image(image)
    # combine the two images
    # note that the order of alpha_composite arguments matters
    overlayed_image = Image.alpha_composite(image, heatmap)
    return overlayed_image