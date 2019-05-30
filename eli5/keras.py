# -*- coding: utf-8 -*-
"""Keras neural network explanations"""

from keras.models import (
    Model, 
    Sequential,
)
from keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import numpy as np

from eli5.base import Explanation
from eli5.explain import explain_prediction


@explain_prediction.register(Model)
def explain_prediction_keras(estimator, doc, # model, image
                             top=None, # not supported
                             top_targets=None, # not supported
                             target_names=None, # rename prediction labels
                             targets=None, # prediction(s) to focus on, if None take top prediction
                             feature_names=None, # not supported
                             feature_re=None, # not supported
                             feature_filter=None, # not supported
                             # new parameters:
                             layers=None, # which layer(s) to focus on,
                             preprocessing=None, # function 
                            ):
    """Explain prediction of a Keras model
    doc : image, 
        one of: path to a single image, directory containing images, 
        PIL image object, or an array can also be multiple images.
    layers: target layer to Grad-CAM on,
        one of: a layer (valid keras layer name (str) or index (int)), a list of layers
    """
    explanation = Explanation(
        repr(estimator), # might want to replace this with something else, eg: estimator.summary()
        description='',
        error='',
        method='gradcam',
        is_regression=False, # classification vs regression model
        highlight_spaces=None, # might be relevant later when explaining text models
    )
    input_dimensions = estimator.input_shape[1:3]
    image = load_image(doc, xDims=input_dimensions, preprocess_fn=preprocessing)

    # TODO: grad-cam on multiple layers by passing a list to layers
    layer_name, layer_index = get_target_layer(layers)

    heatmap = jacobgil(model=estimator, preprocessed_input=image, layer=(layer_name, layer_index))
    heatmap = array_to_img(heatmap)
    image = array_to_img(image[0])
    explanation.heatmap = heatmap
    explanation.image = image
    return explanation

def get_target_layer(layers):
    layer_name = None
    layer_index = None

    if layers is None:
        pass
    elif isinstance(layers, int):
        layer_index = layers
    elif isinstance(layers, str):
        layer_name = layers
    else:
        raise ValueError('Invalid layer (must be str or int): "%s"' % layers)

    return layer_name, layer_index

############ jacobgil's code

def load_image(img_path, xDims=None, preprocess_fn=None):
    # img_path = sys.argv[1]
    # from keras.applications.vgg16 import preprocess_input
    # from keras.applications.xception import preprocess_input # FIXME
    img = load_img(img_path, target_size=xDims)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if preprocess_fn is not None:
        x = preprocess_fn(x)
    # x = next(ImageDataGenerator(rescale=1.0/255).flow(x))
    return x

def jacobgil(model=None, preprocessed_input=None, layer=(None, None)):
    from keras.applications.vgg16 import (
    VGG16, decode_predictions)
    from keras.models import Model
    from keras.preprocessing import image
    from keras.layers.core import Lambda
    from keras.models import Sequential
    from tensorflow.python.framework import ops
    import keras.backend as K
    import tensorflow as tf
    import keras
    import sys
    import cv2

    def target_category_loss(x, category_index, nb_classes):
        return tf.multiply(x, K.one_hot([category_index], nb_classes))

    def target_category_loss_output_shape(input_shape):
        return input_shape

    def normalize(x):
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    def _compute_gradients(tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

    def grad_cam(input_model, image, category_index, layer_name=None, layer_index=None):
        nb_classes = 1000
        target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
        x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
        model = Model(inputs=input_model.input, outputs=x)
        # model.summary() # remove this later
        loss = K.sum(model.output)

        # get the target layer
        # conv_output =  [l for l in model.layers if l.name is layer_name]
        # conv_output = conv_output[0].output

        # TODO: automatically get last Conv layer if layer_name and layer_index are None
        # we need to get the output attribute, else we get a TypeError: Failed to convert object to tensor
        # bottom-up horizontal graph traversal
        layer_index = -4 if layer_index is None else layer_index # FIXME: don't hardcode four
        conv_output = model.get_layer(name=layer_name, index=layer_index).output

        grads = normalize(_compute_gradients(loss, [conv_output])[0])
        gradient_function = K.function([model.input], [conv_output, grads])

        output, grads_val = gradient_function([image])
        output, grads_val = output[0, :], grads_val[0, :, :, :] # FIXME: this probably assumes that the layer is a width*height filter

        weights = np.mean(grads_val, axis = (0, 1))
        # cam = np.ones(output.shape[0 : 2], dtype = np.float32)
        heatmap = np.ones(output.shape[0 : 2], dtype = np.float32)

        for i, w in enumerate(weights):
            # cam += w * output[:, :, i]
            heatmap += w * output[:, :, i]

        heatmap = np.maximum(heatmap, 0) # ReLU
        heatmap = heatmap / np.max(heatmap) # probability
        heatmap = 255*heatmap # 0...255 float
        # we need to insert a channels axis to have an image (channels last by default)
        heatmap = np.expand_dims(heatmap, axis=-1)
        return heatmap

    # preprocessed_input = load_image(img_path)

    # model = VGG16(weights='imagenet')

    predictions = model.predict(preprocessed_input)
    top_1 = decode_predictions(predictions)[0][0]
    print('Predicted class:')
    print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

    predicted_class = np.argmax(predictions)
    heatmap = grad_cam(model, preprocessed_input, predicted_class, layer_name=layer[0], layer_index=layer[1])
    # cv2.imwrite("gradcam.jpg", cam)
    return heatmap

############