# -*- coding: utf-8 -*-
from __future__ import absolute_import
from typing import Union, Optional, Callable

import numpy as np
from PIL import Image
import matplotlib.cm

from eli5.base import Explanation


def format_as_image(expl, # type: Explanation
    resampling_filter=Image.LANCZOS, # type: int
    colormap=matplotlib.cm.viridis, # type: Callable[[np.ndarray], np.ndarray]
    alpha_limit=0.65, # type: Optional[Union[float, int]]
    ):
    # type: (...) -> Image
    """format_as_image(expl, resampling_filter=Image.LANCZOS, colormap=matplotlib.cm.viridis, alpha_limit=0.65)

    Format a :class:`eli5.base.Explanation` object as an image.

    Note that this formatter requires ``matplotlib`` and ``Pillow`` optional dependencies.


    :param Explanation expl:
        :class:`eli5.base.Explanation` object to be formatted.
        It must have an ``image`` attribute with a Pillow image that will be overlaid.
        It must have a ``targets`` attribute, a list of :class:`eli5.base.TargetExplanation` \
        instances that contain the attribute ``heatmap``, \
        a rank 2 numpy array with float values in the interval [0, 1].
        Currently ``targets`` must be length 1 (only one target is supported).


        :raises TypeError: if ``heatmap`` is not a numpy array.
        :raises ValueError: if ``heatmap`` does not contain values as floats in the interval [0, 1].
        :raises TypeError: if ``image`` is not a Pillow image.

    :param resampling_filter:
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
    :type resampling_filter: int, optional

    :param colormap:
        Colormap scheme to be applied when converting the heatmap from grayscale to RGB.
        Either a colormap from matplotlib.cm, 
        or a callable that takes a rank 2 array and 
        returns the colored heatmap as a [0, 1] RGBA numpy array.

        Example colormaps from matplotlib.cm
            * ``viridis``
            * ``jet``
            * ``binary``

        See also https://matplotlib.org/gallery/color/colormap_reference.html.

        Default is ``matplotlib.cm.viridis`` (green/blue to yellow).
    :type colormap: callable, optional

    :param alpha_limit:
        Maximum alpha (transparency / opacity) value allowed 
        for the alpha channel pixels in the RGBA heatmap image.

        Between 0.0 and 1.0.

        Useful when laying the heatmap over the original image, 
        so that the image can be seen over the heatmap.

        Default is 0.65.


        :raises ValueError: if ``alpha_limit`` is outside the [0, 1] interval.
        :raises TypeError: if ``alpha_limit`` is not float, int, or None.
    :type alpha_limit: float or int, optional


    Returns
    -------
    overlay : PIL.Image.Image
        PIL image instance of the heatmap blended over the image.
    """
    image = expl.image
    # validate image
    if not isinstance(image, Image.Image):
        raise TypeError('Explanation image must be a PIL.Image.Image instance. '
                        'Got: {}'.format(image))
    if image.mode != 'RGBA':
        # normalize to 'RGBA'
        image = image.convert('RGBA')

    if not expl.targets:
        # no heatmaps
        return image
    else:
        assert len(expl.targets) == 1
        heatmap = expl.targets[0].heatmap
        _validate_heatmap(heatmap)

    # The order of our operations is: 1. colorize 2. resize
    # as opposed: 1. resize 2. colorize

    # save the original heatmap values
    heatvals = heatmap
    # apply colours to the grayscale array
    heatmap = _colorize(heatmap, colormap=colormap)  # -> rank 3 RGBA array

    # make the alpha intensity correspond to the grayscale heatmap values
    # cap the intensity so that it's not too opaque when near maximum value
    _update_alpha(heatmap, starting_array=heatvals, alpha_limit=alpha_limit)

    heatmap = expand_heatmap(heatmap, image, resampling_filter=resampling_filter)
    overlay = _overlay_heatmap(heatmap, image)
    return overlay


def heatmap_to_image(heatmap):
    # type: (np.ndarray) -> Image
    """
    Convert the numpy array ``heatmap`` to a Pillow image.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Rank 2 grayscale ('L') array or rank 3 coloured ('RGB' or RGBA') array,
        with values in interval [0, 1] as floats.


    :raises TypeError: if ``heatmap`` is not a numpy array.
    :raises ValueError: if ``heatmap`` does not contain values as floats in the interval [0, 1].
    :raises ValueError: if ``heatmap`` rank is neither 2 nor 3.
    :raises ValueError: if rank 3 ``heatmap`` does not have 4 (RGBA) or 3 (RGB) channels.


    Returns
    -------
    heatmap_image : PIL.Image.Image
        Heatmap as an image with a suitable mode.
    """
    _validate_heatmap(heatmap)
    rank = len(heatmap.shape)
    if rank == 2:
        mode = 'L'
    elif rank == 3:
        channels = heatmap.shape[2]
        if channels == 4:
            mode = 'RGBA'
        elif channels == 3:
            mode = 'RGB'
        else:
            raise ValueError('Rank 3 heatmap must have 4 channels (RGBA), '
                             'or 3 channels (RGB). '
                             'Got shape with {} channels'.format(channels))
    else:
        raise ValueError('heatmap must have rank 2 (L, grayscale) ' 
                         'or rank 3 (RGBA, colored). '
                         'Got: %d' % rank)
    heatmap = (heatmap*255).astype('uint8') # -> [0, 255] int
    return Image.fromarray(heatmap, mode=mode)


def _validate_heatmap(heatmap):
    """Check that ``heatmap`` is a numpy array
    with float values between 0 and 1."""
    if not isinstance(heatmap, np.ndarray):
        raise TypeError('heatmap must be a numpy.ndarray instance. '
                        'Got: {}'.format(heatmap))
    mi = np.min(heatmap)
    ma = np.max(heatmap)
    if not (0 <= mi and ma <= 1):
        raise ValueError('heatmap must contain float values '
                         'between 0 and 1 inclusive. '
                         'Got array with minimum: {} ' 
                         'and maximum: {}'.format(mi, ma))


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
    # FIXME: this function may be too specialized and could be refactored
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


def expand_heatmap(heatmap, image, resampling_filter=Image.LANCZOS):
    # type: (np.ndarray, Image, Union[None, int]) -> Image
    """
    Resize the ``heatmap`` image array to fit over the original ``image``,
    using the specified ``resampling_filter`` method.
    The heatmap is converted to an image in the process.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap that is to be resized, as an array.

    image : PIL.Image.Image
        The image whose dimensions will be resized to.

    resampling_filter : int or None
        Interpolation to use when resizing.

        See :func:`eli5.format_as_image` for more details on the `resampling_filter` parameter.


    :raises TypeError: if ``image`` is not a Pillow image instance.


    Returns
    -------
    resized_heatmap : PIL.Image.Image
        The heatmap, resized, as a PIL image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError('image must be a PIL.Image.Image instance. '
                        'Got: {}'.format(image))
    heatmap = heatmap_to_image(heatmap)
    spatial_dimensions = (image.width, image.height)
    heatmap = heatmap.resize(spatial_dimensions, resample=resampling_filter)
    return heatmap


def _overlay_heatmap(heatmap, image):
    # type: (Image, Image) -> Image
    """
    Blend (combine) ``heatmap`` over ``image``, 
    using alpha channel values appropriately (must have mode `RGBA`).
    Output is 'RGBA'.
    """
    # note that the order of alpha_composite arguments matters
    overlayed_image = Image.alpha_composite(image, heatmap)
    return overlayed_image