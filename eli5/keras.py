# -*- coding: utf-8 -*-
"""Keras neural network explanations"""

from keras.models import (
    Model, 
    Sequential,
)

from eli5.base import Explanation
from eli5.explain import explain_prediction


@explain_prediction.register(Model)
def explain_prediction_keras(estimator, doc,
    ):
    """Explain a Keras model prediction"""
    return Explanation(
        repr(estimator), # might want to replace this with something else, eg: estimator.summary()
    )