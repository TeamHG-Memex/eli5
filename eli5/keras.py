# -*- coding: utf-8 -*-
"""Keras neural network explanations"""

import numpy as np
from keras.models import (
    Model, 
    Sequential,
)
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.layers.core import Lambda
import keras.backend as K

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
    doc : image, one of: path to a single image, directory containing images, PIL image object, or an array
    can also be multiple images
    """
    class REPLACEME(Explanation): 
        def __init__(self, *args, REPLACETHISATTR=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.REPLACETHISATTR = REPLACETHISATTR

    explanation = REPLACEME(
        repr(estimator), # might want to replace this with something else, eg: estimator.summary()
        description='',
        error='',
        method='gradcam',
        is_regression=False, # classification vs regression model
        targets=[],
        highlight_spaces=None, # might be relevant later when explaining text models
    )
    # apply grad-cam
    target = targets[0] if isinstance(targets, list) else None
    layer = layer[0] if isinstance(layers, list) else None
    cam = get_grad_cam_explanation(estimator, doc, layer='block5_conv3', target=target, pretrained_weights='imagenet')
    explanation.REPLACETHISATTR = cam
    return explanation


def get_grad_cam_explanation(estimator, doc, layer=None, target=None, pretrained_weights=None):
    # we find out the required dimensions of the image
    # https://stackoverflow.com/questions/43743593/keras-how-to-get-layer-shapes-in-a-sequential-model
    # an alternative is 'estimator.get_layer(index=0).input_shape[1:3]'
    # or 'estimator.get_layer(index=0).output_shape'

    # input_shape = estimator.inputs[0].shape
    # xDims = input_shape[1:3]
    # x = preprocess_example(doc, required_size=xDims, pretrained_weights=pretrained_weights)

    # # get the prediction to focus on
    # if target is None:
    #     target = get_target_prediction(estimator, x, pretrained_weights=pretrained_weights)
    # prediction_index = target[-1]
    # print(prediction_index)

    # get the same estimator but focus on the target prediction
    # output_units = estimator.output.shape[-1]
    # filter_output = Lambda(lambda x: x*K.one_hot(prediction_index, output_units), output_shape=lambda x: x)
    # forward_predictor = Model(inputs=estimator.inputs[0].shape, outputs=filter_output)

    ##### jacobgil's code
    import tensorflow as tf
    import sys
    from keras.applications.imagenet_utils import preprocess_input, decode_predictions

    def load_image(path):
        img = load_img(path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    x = load_image(doc)
    predictions = estimator.predict(x)
    top_1 = decode_predictions(predictions)[0][0]
    prediction_index = np.argmax(predictions)

    def _compute_gradients(tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var) 
            for var, grad in zip(var_list, grads)]

    def target_category_loss(x, prediction_index, nb_classes):
        return tf.multiply(x, K.one_hot([prediction_index], nb_classes))

    def target_category_loss_output_shape(input_shape):
        return input_shape

    def normalize(x):
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    nb_classes = 1000 # number of classes

    target_layer = lambda x: target_category_loss(x, prediction_index, nb_classes)
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(estimator.output)
    model = Model(inputs=estimator.input, outputs=x)
    model.summary() # print layers, number of parameters
    
    loss = K.sum(model.output) # sum values of output tensor
    conv_output =  [l for l in model.layers if l.name is layer][0].output
    
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([x])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))

    cam = np.maximum(cam, 0)

    heatmap = cam / np.max(cam)
 
    x = x[0, :] # 4d -> 3d
    x -= np.min(x) # -(-smallest) = add to x (make positive)
    x = np.minimum(x, 255) # cap at 255

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(x)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap3

    ####

    cam = array_to_img(x[0])
    return cam


def preprocess_example(doc, required_size=None, pretrained_weights=None):
    # assuming doc is a single input sample
    # required_size is a tuple of (height, width)

    # we need to resize the height and witdth of our image to suit the model
    # https://stackoverflow.com/questions/43017017/keras-model-predict-for-a-single-image
    img = load_img(doc, target_size=required_size)

    # convert a PIL object to a numpy array
    imgarr = img_to_array(img)

    # our current shape is (height, width, depth)
    # we need to insert an axis at the 0th position to indicate the batch size
    # this is required by the keras predict() function
    x = np.expand_dims(imgarr, axis=0) # get (batch size, height, width, depth)

    if pretrained_weights == 'imagenet':
        from keras.applications.imagenet_utils import preprocess_input
        x = preprocess_input(x)
        # FIXME: this makes the image blue!

    return x


def get_target_prediction(estimator, x, pretrained_weights=None):
    # get the prediction manually
    y = estimator.predict(x)

    # get the first maximum probability index
    index = y.argmax(axis=-1)[0] 

    # we decode the predicted class
    if pretrained_weights == 'imagenet':
        from keras.applications.imagenet_utils import decode_predictions
        # predictions -> top 1
        top_prediction = decode_predictions(y, top=1)[0][0]
        cls_name, cls_descr, score = top_prediction
    else:
        cls_name = ''
        cls_descr = ''
        score = y[0, index]
    return cls_name, cls_descr, score, index



### hard


def get_prediction_forward_activation():
    pass


def get_target_layer_forward_activation():
    pass


def get_backward_gradients():
    pass


### easy


def get_activation_weights():
    pass


def get_localization_map():
    pass


def global_avg_pool():
    pass


def relu():
    pass