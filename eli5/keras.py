# -*- coding: utf-8 -*-
"""Keras neural network explanations"""

from keras.models import (
    Model, 
    Sequential,
)
from keras.preprocessing.image import load_img, img_to_array, array_to_img

from eli5.base import Explanation, TargetExplanation
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
                             layers=None, # which layer(s) to focus on
                            ):
    """Explain prediction of a Keras model
    doc : image, 
        one of: path to a single image, directory containing images, 
        PIL image object, or an array can also be multiple images.
    """
    explanation = Explanation(
        repr(estimator), # might want to replace this with something else, eg: estimator.summary()
        description='',
        error='',
        method='gradcam',
        is_regression=False, # classification vs regression model
        targets=[],
        highlight_spaces=None, # might be relevant later when explaining text models
    )
    cam, heatmap = jacobgil(model=estimator, img_path=doc)
    cam = array_to_img(cam)
    explanation.heatmap = cam
    return explanation

############ jacobgil's code

def jacobgil(model=None, img_path=None):
    from keras.applications.vgg16 import ( # pre-trained network
    VGG16, preprocess_input, decode_predictions)
    from keras.models import Model # functional API
    from keras.preprocessing import image # image preprocessing
    from keras.layers.core import Lambda # lambda expression as a layer
    from keras.models import Sequential # sequential API
    from tensorflow.python.framework import ops # internal API 
    import keras.backend as K # abstract backend API for keras
    import tensorflow as tf
    import numpy as np
    import keras
    import sys
    import cv2

    def target_category_loss(x, category_index, nb_classes):
        return tf.multiply(x, K.one_hot([category_index], nb_classes))

    def target_category_loss_output_shape(input_shape):
        return input_shape

    def normalize(x):
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    def load_image(img_path):
        # img_path = sys.argv[1]
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def _compute_gradients(tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

    def grad_cam(input_model, image, category_index, layer_name):
        nb_classes = 1000
        target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
        x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
        model = Model(inputs=input_model.input, outputs=x)
        model.summary()
        loss = K.sum(model.output)
        conv_output =  [l for l in model.layers if l.name is layer_name][0].output
        grads = normalize(_compute_gradients(loss, [conv_output])[0])
        gradient_function = K.function([model.input], [conv_output, grads])

        output, grads_val = gradient_function([image])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        weights = np.mean(grads_val, axis = (0, 1))
        cam = np.ones(output.shape[0 : 2], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        heatmap = cam / np.max(cam)

        #Return to BGR [0..255] from the preprocessed image
        image = image[0, :]
        image -= np.min(image)
        image = np.minimum(image, 255)

        cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        cam = np.float32(cam) + np.float32(image)
        cam = 255 * cam / np.max(cam)
        return np.uint8(cam), heatmap

    preprocessed_input = load_image(img_path)

    # model = VGG16(weights='imagenet')

    predictions = model.predict(preprocessed_input)
    top_1 = decode_predictions(predictions)[0][0]
    print('Predicted class:')
    print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

    predicted_class = np.argmax(predictions)
    cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")
    # cv2.imwrite("gradcam.jpg", cam)
    return cam, heatmap

############