# -*- coding: utf-8 -*-

import PIL
import matplotlib.pyplot as plt
import matplotlib.cm


def format_as_image(expl):
    import cv2
    from keras.preprocessing.image import img_to_array, array_to_img
    import numpy as np

    heatmap = expl.heatmap
    image = expl.image

    image = img_to_array(image)

    # heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = heatmap.resize((224, 224), resample=PIL.Image.LANCZOS)

    heatmap = img_to_array(heatmap)

    # image -= np.min(image)
    # image = np.minimum(image, 255)

    heatmap = np.uint64(heatmap)

    # threshold
    heatmap[heatmap < 100] = 0
    # consider setting transparency to 0, not the pixel itself

    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # heatmap = np.expand_dims(heatmap, axis=-1)

    fig, ax = plt.subplots()
    ax.axis('off')
    width, height = image.shape[:2]
    extent = [0, width, 0, height]
    I = ax.imshow(image[:,:,0], extent=extent)
    H = ax.contourf(heatmap[:,:,0], alpha=0.2, extent=extent, cmap='jet')
    
    # heatmap = np.uint8(H.get_cmap()(heatmap[:,:,0] / 255))[:,:,:3]
    overlayed_image = np.float32(heatmap) + np.float32(image)
    overlayed_image = 255 * overlayed_image / np.max(overlayed_image)
    # fig, ax = plt.subplots()
    # ax.imshow(overlayed_image[:,:,0], interpolation='nearest', alpha=1, extent=extent)
    # plt.show()

    return array_to_img(overlayed_image)