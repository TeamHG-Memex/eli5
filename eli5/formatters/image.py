# -*- coding: utf-8 -*-
from typing import Union, Optional, Callable

import numpy as np # type: ignore
from PIL import Image # type: ignore
import matplotlib.cm # type: ignore

from eli5.base import Explanation


def format_as_image(expl, # type: Explanation
    interpolation=Image.LANCZOS, # type: int
    colormap=matplotlib.cm.magma, # type: Callable[[np.ndarray], np.ndarray]
    alpha_limit=0.65, # type: Optional[Union[float, int]]
    ):
    # type: (...) -> Image
    """format_as_image(expl, interpolation=Image.LANCZOS, colormap=matplotlib.cm.magma, alpha_limit=0.65)

    Format a :class:`eli5.base.Explanation` object as an image.

    Note that this formatter requires ``matplotlib`` and ``Pillow`` optional dependencies.
    
    
    :param interpolation `int, optional`:
        Interpolation ID or Pillow filter to use when resizing the image.
        
        Example filters from PIL.Image
            * ``NEAREST``
            * ``BOX``
            * ``BILINEAR``
            * ``HAMMING``
            * ``BICUBIC``
            * ``LANCZOS``

        See also `<https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters>`_.

        *Note that these attributes are integer values*.

        Default is ``PIL.Image.LANCZOS``.


    :param colormap `callable, optional`:
        Colormap scheme to be applied when converting the heatmap from grayscale to RGB.
        Either a colormap from matplotlib.cm, 
        or a callable that takes a rank 2 array and 
        returns the colored heatmap as a [0, 1] RGBA numpy array.

        Example colormaps from matplotlib.cm
            * ``viridis``
            * ``jet``
            * ``binary``
        
        See also https://matplotlib.org/gallery/color/colormap_reference.html.

        Default is ``matplotlib.cm.magma`` (blue to red).



    :param alpha_limit `float or int, optional`:
        Maximum alpha (transparency / opacity) value allowed 
        for the alpha channel pixels in the RGBA heatmap image.

        Between 0.0 and 1.0.

        Useful when laying the heatmap over the original image, 
        so that the image can be seen over the heatmap.

        Default is 0.65.


        :raises ValueError: if ``alpha_limit`` is outside the [0, 1] interval.
        :raises TypeError: if ``alpha_limit`` is not float, int, or None.


    Returns
    -------
    overlay : PIL.Image.Image
        PIL image instance of the heatmap blended over the image.
    """
    image = expl.image
    heatmap = expl.heatmap
    
    # We first 1. colorize 2. resize
    # as opposed 1. resize 2. colorize

    heatmap = _colorize(heatmap, colormap=colormap) # -> rank 3 RGBA array
    # TODO: automatically detect which colormap would be the best based on colors in the image

    # make the alpha intensity correspond to the grayscale heatmap values
    # cap the intensity so that it's not too opaque when near maximum value
    # TODO: more options for controlling alpha, i.e. a callable?
    heat_values = expl.heatmap
    _update_alpha(heatmap, starting_array=heat_values, alpha_limit=alpha_limit)

    heatmap = expand_heatmap(heatmap, image, interpolation=interpolation)
    overlay = overlay_heatmap(heatmap, image)
    return overlay


def heatmap_to_grayscale(heatmap):
    # type: (np.ndarray) -> Image
    """
    Convert ``heatmap`` array into a grayscale PIL image.
    
    Parameters
    ----------
    heatmap: numpy.ndarray
        a rank 2 (2D) numpy array with [0, 1] float values.

    Returns
    -------
    heatmap_img : PIL.Image.Image
        A grayscale (mode 'L') PIL Image.
    """
    heatmap = (heatmap*255).astype('uint8') # -> [0, 255] int
    return Image.fromarray(heatmap, 'L') # -> grayscale PIL


def heatmap_to_rgba(heatmap):
    # type: (np.ndarray) -> Image
    """
    Convert ``heatmap`` to an RGBA PIL image.

    Parameters
    ----------
    heatmap : PIL.Image.Image
        A rank 2 (2D) numpy array with [0, 1] float values.

    Returns
    -------
    heatmap_img : PIL.Image.Image
        A coloured, alpha-channel (mode 'RGBA') PIL Image.
    """
    heatmap = (heatmap*255).astype('uint8') # -> [0, 255] int
    return Image.fromarray(heatmap, 'RGBA') # -> RGBA PIL


def _colorize(heatmap, colormap):
    # type: (np.ndarray, Callable[[np.ndarray], np.ndarray]) -> np.ndarray
    """
    Apply the ``colormap`` function to a grayscale 
    rank 2 ``heatmap`` array (with float values in interval [0, 1]).
    Returns an RGBA rank 3 array with float values in range [0, 1].
    """
    heatmap = colormap(heatmap) # -> [0, 1] RGBA ndarray
    return heatmap


def _update_alpha(image_array, starting_array=None, alpha_limit=None):
    # type: (np.ndarray, Optional[np.ndarray], Optional[Union[float, int]]) -> None
    """
    Update the alpha channel values of an RGBA rank 3 ndarray ``image_array``,
    optionally creating the alpha channel from rank 2 ``starting_array``, 
    and setting upper limit for alpha values (opacity) to ``alpha_limit``.    

    This function modifies ``image_array`` in-place.
    """
    # get the alpha channel slice
    if isinstance(starting_array, np.ndarray):
        alpha = starting_array
    else:
        # take the alpha channel as is
        alpha = image_array[:,:,3]
    # set maximum alpha value
    alpha = _cap_alpha(alpha, alpha_limit)
    # update alpha channel in the original image
    image_array[:,:,3] = alpha
    # TODO: optimisation?


def _cap_alpha(alpha_arr, alpha_limit):
    # type: (np.ndarray, Union[None, float, int]) -> np.ndarray
    """
    Limit the alpha values in ``alpha_arr``
    by setting the maximum alpha value to ``alpha_limit``.
    Returns a a new array with the values capped.
    """
    if alpha_limit is None:
        return alpha_arr
    elif isinstance(alpha_limit, (float, int)):
        if 0 <= alpha_limit <= 1:
            new_alpha = np.minimum(alpha_arr, alpha_limit)
            return new_alpha
        else:
            raise ValueError('alpha_limit must be' 
                             'between 0 and 1 inclusive, got: %f' % alpha_limit)
    else:
        raise TypeError('alpha_limit must be int or float,' 
                        'got: {}'.format(alpha_limit))


def expand_heatmap(heatmap, image, interpolation):
    # type: (np.ndarray, Image, Union[None, int]) -> Image
    """ 
    Resize the ``heatmap`` image array to fit over the original ``image``,
    using the specified ``interpolation`` method.
    
    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap that is to be resized, a rank 2 (grayscale) or rank 3 (colored) array.

    image : PIL.Image.Image
        The image whose dimensions will be resized to.

    interpolation : int or None
        Interpolation to use when resizing.

        See :func:`eli5.format_as_image` for more details on the `interpolation` parameter.


    :raises ValueError: if heatmap's dimensions are not rank 2 or rank 3.


    Returns
    -------
    resized_image : PIL.Image.Image
        A resized PIL image.
    """
    rank = len(heatmap.shape)
    if rank == 2:
        heatmap = heatmap_to_grayscale(heatmap)
    elif rank == 3:
        heatmap = heatmap_to_rgba(heatmap)
    else:
        raise ValueError('heatmap array must have rank 2 (L, grayscale)' 
                         'or rank 3 (RGBA, colored).'
                         'Got: %d' % rank)
    # PIL seems to have a much nicer API for resizing than scipy (scipy.ndimage)
    # Also, scipy seems to have some interpolation problems: 
    # https://github.com/scipy/scipy/issues/8210
    spatial_dimensions = (image.width, image.height)
    heatmap = heatmap.resize(spatial_dimensions, resample=interpolation)
    return heatmap
    # TODO: resize a numpy array without converting to PIL image?


def _convert_image(img):
    # type: (Union[np.ndarray, Image]) -> Image
    """ 
    Convert the ``img`` numpy array or PIL Image (any mode)
    to an RGBA PIL Image.
    
    :raises TypeError: if ``img`` is neither a numpy.ndarray or PIL.Image.Image.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img) # ndarray -> PIL image
    if isinstance(img, Image.Image):
        img = img.convert(mode='RGBA') # -> RGBA image
    else:
        raise TypeError('img must be numpy.ndarray or PIL.Image.Image'
                        'got: {}'.format(img))
    return img


def overlay_heatmap(heatmap, image):
    # type: (Image, Image) -> Image
    """
    Blend ``heatmap`` over ``image``, 
    using alpha channel values appropriately.

    Parameters
    ----------
    heatmap : PIL.Image.Image
        The heatmap image, mode 'RGBA'.

    image: PIL.Image.Image
        The original image, mode 'RGBA'.
    
    Returns
    -------
    overlayed_image : PIL.Image.Image
        A blended PIL image, mode 'RGBA'.
    """
    # normalise to same format
    heatmap = _convert_image(heatmap)
    image = _convert_image(image)
    # combine the two images
    # note that the order of alpha_composite arguments matters
    overlayed_image = Image.alpha_composite(image, heatmap)
    return overlayed_image
