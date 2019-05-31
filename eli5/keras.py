# -*- coding: utf-8 -*-
"""Keras neural network explanations"""

from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import numpy as np

from eli5.base import Explanation
from eli5.explain import explain_prediction

# note that Sequential subclasses Model, so we can just register the Model type
# Model subclasses Network, but is using Network with this function valid?
@explain_prediction.register(Model) 
def explain_prediction_keras(estimator, doc, # model, image
                             top=None, # NOT SUPPORTED
                             top_targets=None, # NOT SUPPORTED
                             target_names=None, # rename / provide prediction labels
                             targets=None, # prediction(s) to focus on, if None take top prediction
                             feature_names=None, # NOT SUPPORTED
                             feature_re=None, # NOT SUPPORTED
                             feature_filter=None, # NOT SUPPORTED
                             # new parameters:
                             layer=None, # which layer to focus on, 
                             prediction_decoder=None, # target prediction decoding function
                            ):
    """Explain prediction of a Keras model
    doc : image, 
        one of: must be an input acceptable by estimator,
        preprocessing can be done with helper functions.
    targets: predictions
        a list of predictions
        integer for ImageNet classification
    layer: valid target layer in the model to Grad-CAM on,
        one of: a valid keras layer name (str) or index (int), 
        a callable function that returns True when the desired layer is matched for the model
        if None, automatically use a helper callable function to get the last suitable Conv layer
    """
    explanation = Explanation(
        repr(estimator), # might want to replace this with something else, eg: estimator.summary()
        description='',
        error='',
        method='gradcam',
        is_regression=False, # classification vs regression model
        highlight_spaces=None, # might be relevant later when explaining text models
    )

    # TODO: grad-cam on multiple layers by passing a list of layers
    if layer is None:
        # Automatically get the layer if not provided
        # this might not be a good idea from transparency / user point of view
        layer = get_last_activation_maps
    target_layer = get_target_layer(estimator, layer)

    # get prediction to focus on
    if targets is None:
        predicted = get_target_prediction(estimator, doc, decoder=prediction_decoder)
    else:
        predicted = targets[0] 
        # TODO: take in a single target as well, not just a list
        # does it make sense to take a list of targets. You can only Grad-CAM a single target?
        # TODO: consider changing signature / types for explain_prediction generic function
        # TODO: need to find a way to show the label for the passed prediction as well as its probability

    heatmap = jacobgil(estimator, doc, predicted, target_layer)
    heatmap = array_to_img(heatmap)
    image = array_to_img(doc[0])
    explanation.heatmap = heatmap
    explanation.image = image
    return explanation


def get_target_prediction(model, x, decoder=None):
    predictions = model.predict(x)
    if decoder is not None:
        # TODO: check if decoder is callable?
        # FIXME: it is not certain that we need such indexing into the decoder's output
        top_1 = decoder(predictions)[0][0] 
        ncode, label, proba = top_1
        # TODO: do I print, log, or append to 'description' this output?
        print('Predicted class:') 
        print('%s (%s) with probability %.2f' % (label, ncode, proba))
    # FIXME: non-classification tasks
    predicted_class = np.argmax(predictions)
    return predicted_class


def get_target_layer(estimator, desired_layer):
    # Return instance of the desired layer in the model

    if isinstance(desired_layer, int):
        # conv_output =  [l for l in model.layers if l.name is layer_name]
        # conv_output = conv_output[0]
        # bottom-up horizontal graph traversal
        target_layer = estimator.get_layer(index=desired_layer)
        # These can raise ValueError if the layer index / name specified is not found
    elif isinstance(desired_layer, str):
        target_layer = estimator.get_layer(name=desired_layer)
    elif callable(desired_layer):
        # is 'callable' the right function to use here?
        target_layer = desired_layer(estimator)
    else:
        raise ValueError('Invalid desired_layer (must be str, int, or callable): "%s"' % desired_layer)

    # TODO: check target_layer dimensions (is it possible to do Grad-CAM on it?)
    return target_layer


def get_last_activation_maps(estimator):
    # TODO: automatically get last Conv layer if layer_name and layer_index are None
    # FIXME: don't hardcode four
    # Some ideas:
    # 1. linear search backwards (using negative indexing with get_layer()) until find the desired layer
    # 2. look at layer name, exclude things like "softmax", "global average pooling", 
    # etc, include things like "conv" (but watch for false positives)
    # 3. look at layer input/output dimensions, to ensure they match
    # 4. If can't find, either raise error, or try some layer and log a warning
    target_layer = estimator.get_layer(index=-4)
    return target_layer


############ jacobgil's code


def preprocess_image(img, estimator=None, preprocessing=None):
    # path to a single image, directory containing images, 
    # PIL image object, or an array can also be multiple images.
    # preprocessing function is an optional callable
    xDims = None
    if estimator is not None:
        xDims = estimator.input_shape[1:3]
    im = load_img(img, target_size=xDims)
    x = img_to_array(im)
    x = np.expand_dims(x, axis=0)
    if preprocessing is not None:
        # eg:
        # from keras.applications.vgg16 import preprocess_input
        # from keras.applications.xception import preprocess_input # FIXME
        x = preprocessing(x)
    # x = next(ImageDataGenerator(rescale=1.0/255).flow(x))
    return x

def jacobgil(model, preprocessed_input, predicted_class, layer):
    from keras.models import Model
    from keras.preprocessing import image
    from keras.layers.core import Lambda
    from keras.models import Sequential
    from tensorflow.python.framework import ops
    import keras.backend as K
    import tensorflow as tf
    import keras
    import sys

    def target_category_loss(x, category_index, nb_classes):
        return tf.multiply(x, K.one_hot([category_index], nb_classes))

    def target_category_loss_output_shape(input_shape):
        return input_shape

    def normalize(x):
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    def _compute_gradients(tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

    def grad_cam(input_model, image, category_index, layer):
        # FIXME: this assumes that we are doing classification
        # nb_classes = 1000 # FIXME: number of classes can be variable
        nb_classes = input_model.output_shape[1] # TODO: test this

        # FIXME: rename these "layer" variables
        target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
        x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
        model = Model(inputs=input_model.input, outputs=x)
        # model.summary() # remove this later
        loss = K.sum(model.output)

        # we need to get the output attribute, else we get a TypeError: Failed to convert object to tensor
        conv_output = layer.output

        grads = normalize(_compute_gradients(loss, [conv_output])[0])
        gradient_function = K.function([model.input], [conv_output, grads])

        output, grads_val = gradient_function([image]) # work happens here
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

    heatmap = grad_cam(model, preprocessed_input, predicted_class, layer)
    # cv2.imwrite("gradcam.jpg", cam)
    return heatmap

############