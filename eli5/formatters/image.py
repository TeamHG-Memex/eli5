# -*- coding: utf-8 -*-

import PIL
import matplotlib.pyplot as plt
import matplotlib.cm
from keras.preprocessing.image import img_to_array, array_to_img
import numpy as np


def format_as_image(expl):
    image = expl.image
    heatmap = expl.heatmap

    image = img_to_array(image)
    # image -= np.min(image)
    # image = np.minimum(image, 255)
    image = np.uint8(image)

    # use Pillow to resize the heatmap to be the size of the image
    image_dimensions = image.shape[:2]
    heatmap = heatmap.resize(image_dimensions, resample=PIL.Image.LANCZOS)
    # heatmap = cv2.resize(heatmap, (224, 224))

    heatmap = img_to_array(heatmap)
    heatmap = np.uint8(heatmap)

    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # heatmap = np.expand_dims(heatmap, axis=-1)

    # scale to [0, 1]
    heatmap = np.float32(heatmap / 255)
    image = np.float32(image / 255)

    # apply colour map
    heatmap_grayscale = heatmap[:,:,0] # save grayscale width*height version
    heatmap = np.float32(matplotlib.cm.jet(heatmap[:,:,0]))

    # insert alpha channel
    image = np.dstack((image, np.ones(image_dimensions, dtype=np.float32)))
    # print(heatmap.shape, heatmap.dtype, np.min(heatmap), np.max(heatmap))

    # modify alpha
    # 1.
    # heatmap = np.where(heatmap < 0.85, 0*heatmap, heatmap) # too dark
    # heatmap = np.where(heatmap < 0.85, image, heatmap) # too bright
    # 2.
    # heatmap_alpha = heatmap[:,:,3]
    # threshold = 0.1
    # heatmap_alpha[heatmap_grayscale < threshold] = 0.0
    # heatmap_alpha[threshold < heatmap_grayscale] = 0.5
    # heatmap[:,:,3] = heatmap_alpha
    # 3. FIXES: make the transition from no heatmap to heatmap less sharp
    # alpha intensity corresponds / is proportional to grayscale value intensity
    heatmap_alpha = heatmap_grayscale
    # cap the intensity so that it's not too bright when near 1
    cap = 0.65
    heatmap_alpha = np.minimum(heatmap_alpha, cap) # choose the lower of two values
    heatmap[:,:,3] = heatmap_alpha # update alpha channel
    # TODO: efficiency of this approach?

    fig, ax = plt.subplots()
    ax.axis('off')
    # width, height = image.shape[:2]
    # extent = [0, width, 0, height]
    # print(image.shape, image.dtype, np.min(image), np.max(image))

    I = ax.imshow(image)
    H = ax.imshow(heatmap)
    # H = ax.contourf(heatmap[:,:,0], alpha=0.5, clim=[100, 255])

    plt.show()
    # FIXME: for some reason image is resized to 231x231 from 299x299
    # FIXME: sometimes the plot doesn't show up, only get a 'Axis for figure n' message

    # overlayed_image = np.float32(heatmap) + np.float32(image)
    # overlayed_image = 255 * overlayed_image / np.max(overlayed_image)
    # overlayed_image = array_to_img(overlayed_image)

    image = array_to_img(image)
    heatmap = array_to_img(heatmap)
    overlayed_image = PIL.Image.alpha_composite(image, heatmap)

    return overlayed_image