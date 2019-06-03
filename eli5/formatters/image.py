# -*- coding: utf-8 -*-

import numpy as np
import PIL
import matplotlib.pyplot as plt
import matplotlib.cm


def format_as_image(expl):
    # Get PIL Image instances
    image = expl.image
    heatmap = expl.heatmap
    
    heatmap = resize_over(heatmap, image) # resize the heatmap to be the same as the image
    heatmap = np.array(heatmap) # PIL image -> ndarray spatial map, [0, 255]
    heatmap_grayscale = heatmap # save the 'pre-colour' (grayscale) heatmap
    heatmap = colourise(heatmap) # apply colour map

    # update the alpha channel (transparency/opacity) values of the heatmap
    # make the alpha intensity correspond to the grayscale heatmap values
    # cap the intensity so that it's not too opaque when near maximum value
    heatmap = set_alpha(heatmap, starting_array=heatmap_grayscale, cap_value=165.75)

    overlay = overlay_heatmap(heatmap, image)
    show_interactive(overlay, expl)

    return overlay


def show_interactive(overlay, expl):
    """Show the overlayed heatmap over image in a matplotlib plot"""
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
    return (image.width, image.height)


def resize_over(heatmap, image, interpolation=PIL.Image.LANCZOS):
    """Resize the `heatmap` image to fit over the original `image`,
    optionally using an `interpolation` algorithm as a filter from PIL.Image"""
    heatmap = heatmap.resize(get_spatial_dimensions(image), resample=interpolation)
    return heatmap


def colourise(heatmap):
    """Apply colour to a grayscale heatmap, returning an RGBA [0, 255] ndarray"""
    heatmap = matplotlib.cm.jet(heatmap) # -> [0, 1] RGBA ndarray
    heatmap = np.uint8(heatmap*255) # re-scale: [0, 1] -> [0, 255] ndarray
    return heatmap


def set_alpha(image_array, starting_array=None, cap_value=None):
    """Update alpha channel values of an RGBA ndarray `image_array`,
    optionally creating the alpha channel from `starting_array`
    and setting upper limit for alpha values (opacity) to `cap_value`"""
    if isinstance(starting_array, np.ndarray):
        alpha = starting_array
    else:
        alpha = image_array[:,:,3]
    if cap_value is not None:
        alpha = np.minimum(alpha, cap_value)
    image_array[:,:,3] = alpha
    return image_array
    # Efficiency of this approach?


def normalise_image(img):
    """Convert an np.ndarray or PIL Image instance to an RGBA PIL Image"""
    if isinstance(img, np.ndarray):
        img = PIL.Image.fromarray(img)
    if isinstance(img, PIL.Image.Image):
        if img.mode == 'RGB':
            img = img.convert(mode='RGBA') # RGB image -> RGBA image
    return img


def overlay_heatmap(heatmap, image):
    """Overlay 'heatmap' over 'image'"""
    # perform normalisation steps
    heatmap = normalise_image(heatmap)
    image = normalise_image(image)
    # combine the two images
    overlayed_image = PIL.Image.alpha_composite(image, heatmap) # the order of arguments matters!
    return overlayed_image