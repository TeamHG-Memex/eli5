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
    image = np.uint8(image)

    # heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = heatmap.resize((224, 224), resample=PIL.Image.LANCZOS)

    heatmap = img_to_array(heatmap)

    # image -= np.min(image)
    # image = np.minimum(image, 255)

    heatmap = np.uint8(heatmap)


    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # heatmap = np.expand_dims(heatmap, axis=-1)

    # convert to [0, 1]
    heatmap = np.float32(heatmap / 255)
    image = np.float32(image / 255)

    # apply colour map
    heatmap = np.float32(matplotlib.cm.jet(heatmap[:,:,0]))

    # insert alpha channel
    image = np.dstack((image, np.ones((224, 224), dtype=np.float32)))
    # print(heatmap.shape, heatmap.dtype, np.min(heatmap), np.max(heatmap))

    # threshold
    # colour = heatmap[:,:]
    # colour = colour < 0.5
    # print(colour)
    # mask = np.all(heatmap == colour, axis=2)
    # heatmap[mask] = [0,0,0,0]
    heatmap = np.where(heatmap < 0.85, 0*heatmap, heatmap)
    # mask = heatmap[:,:] < 0.5
    # heatmap[mask] = [0,0,0,0]
    # heatmap[heatmap[:,:] < 100] = 0

    fig, ax = plt.subplots()
    ax.axis('off')
    # width, height = image.shape[:2]
    # extent = [0, width, 0, height]
    # print(image.shape, image.dtype, np.min(image), np.max(image))
    # print(heatmap.shape, heatmap.dtype, np.min(heatmap), np.max(heatmap))
    
    I = ax.imshow(image)
    # H = ax.contourf(heatmap[:,:,0], alpha=0.5, cmap='jet', clim=[100, 255])
    H = ax.imshow(heatmap, alpha=0.6)

    overlayed_image = np.float32(heatmap) + np.float32(image)
    overlayed_image = 255 * overlayed_image / np.max(overlayed_image)
    # fig, ax = plt.subplots()
    # ax.imshow(overlayed_image[:,:,0], interpolation='nearest', alpha=1, extent=extent)
    plt.show()

    return array_to_img(overlayed_image)