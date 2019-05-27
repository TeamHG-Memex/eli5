# -*- coding: utf-8 -*-
"""Keras neural network explanations"""

from keras.models import (
    Model, 
    Sequential,
)
from keras.preprocessing.image import load_img, array_to_img

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
    doc : image, one of path to an image, directory containing images, PIL image object, or an array
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
    # apply grad-cam
    target = targets[0] if isinstance(targets, list) else None
    cam = get_grad_cam_explanation(estimator, doc, target, layers)
    explanation.targets.append(cam)
    return explanation


def get_grad_cam_explanation(estimator, doc, target, layers):
    imgarr = load_img(doc)
    cam = array_to_img(imgarr)
    return cam