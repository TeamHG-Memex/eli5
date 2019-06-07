# -*- coding: utf-8 -*-

import numpy as np
import PIL
import matplotlib.pyplot as plt
import matplotlib.cm


def format_as_image(expl,
    interactive=False, # remove this later
    interpolation=PIL.Image.LANCZOS,
    colormap=matplotlib.cm.magma,
    alpha_limit=165.75,
    ):
    """ Format explanation as an image.
    
    Parameters
    ----------
    interpolation: int, optional
        Interpolation ID / PIL filter to be used when resizing the image.
        Default is PIL.Image.LANCZOS.
    colormap: matplotlib colormap object, optional
        Colormap scheme to be applied when converting the heatmap from grayscale to RGB.
        Default is matplotlib.cm.magma.
    alpha_limit: float (0. to 255.), optional
        Maximum alpha (transparency / opacity) value allowed 
        for the alpha channel in the RGBA heatmap image.
        Useful when laying the heatmap over the original image, 
        so that the image can be seen over the heatmap.
        Default is alpha_limit=165.75.
    """
    # Get PIL Image instances
    image = expl.image
    heatmap = expl.heatmap
    
    heatmap = resize_over(heatmap, image, interpolation=interpolation) # resize the heatmap to be the same as the image
    heatmap = np.array(heatmap) # PIL image -> ndarray spatial map, [0, 255]
    heatmap_grayscale = heatmap # save the 'pre-colour' (grayscale) heatmap
    heatmap = colourise(heatmap, colormap=colormap) # apply colour map

    # update the alpha channel (transparency/opacity) values of the heatmap
    # make the alpha intensity correspond to the grayscale heatmap values
    # cap the intensity so that it's not too opaque when near maximum value
    heatmap = set_alpha(heatmap, starting_array=heatmap_grayscale, alpha_limit=alpha_limit)

    overlay = overlay_heatmap(heatmap, image)
    if interactive:
        show_interactive(overlay, expl)

    return overlay


def show_interactive(overlay, expl):
    """Show the overlayed heatmap over image in a matplotlib plot (to be moved to show_prediction)"""
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.margins(0)
    plt.title(expl.estimator) # TODO: add image, prediction, layer information as well

    ax.imshow(overlay)
    plt.show()
    return fig, ax

    # FIXME: for some reason image is resized to 231x231 from 299x299
    # FIXME: sometimes the plot doesn't show up, only get a 'Axis for figure n' message


def get_spatial_dimensions(image):
    return (image.height, image.width)


def resize_over(heatmap, image, interpolation):
    """Resize the `heatmap` image to fit over the original `image`,
    using the specified `interpolation`"""
    # TODO: try scipy.ndimage.interpolation
    heatmap = heatmap.resize(get_spatial_dimensions(image), resample=interpolation)
    return heatmap


def colourise(heatmap, colormap):
    """Apply colour to a grayscale heatmap, returning an RGBA [0, 255] ndarray"""
    # TODO: take colormap as callable
    heatmap = colormap(heatmap) # -> [0, 1] RGBA ndarray
    heatmap = np.uint8(heatmap*255) # re-scale: [0, 1] -> [0, 255] ndarray
    return heatmap
    # TODO: be able to choose which heatmap to apply


def set_alpha(image_array, starting_array=None, alpha_limit=None):
    """Update alpha channel values of an RGBA ndarray `image_array`,
    optionally creating the alpha channel from `starting_array`
    and setting upper limit for alpha values (opacity) to `alpha_limit`"""
    if isinstance(starting_array, np.ndarray):
        alpha = starting_array
    else:
        alpha = image_array[:,:,3]
    if alpha_limit is not None:
        alpha = np.minimum(alpha, alpha_limit)
    image_array[:,:,3] = alpha
    return image_array
    # Efficiency of this approach?


def convert_image(img):
    """Convert an np.ndarray or PIL.Image.Image instance to an RGBA PIL Image"""
    if isinstance(img, np.ndarray):
        img = PIL.Image.fromarray(img) # ndarray -> PIL image
    if isinstance(img, PIL.Image.Image):
        if img.mode == 'RGB':
            img = img.convert(mode='RGBA') # RGB image -> RGBA image
    return img


def overlay_heatmap(heatmap, image):
    """Overlay 'heatmap' over 'image'"""
    # perform normalisation steps
    heatmap = convert_image(heatmap)
    image = convert_image(image)
    # combine the two images
    overlayed_image = PIL.Image.alpha_composite(image, heatmap) # the order of arguments matters!
    return overlayed_image