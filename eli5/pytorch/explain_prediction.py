# -*- coding: utf-8 -*-

import numpy as np # type: ignore
import torch # type: ignore
from torch.nn import Module # type: ignore

from eli5.explain import explain_prediction


@explain_prediction.register(Module)
def explain_prediction_pytorch(module, doc, image=None):
    print('here')