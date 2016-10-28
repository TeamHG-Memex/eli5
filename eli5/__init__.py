# -*- coding: utf-8 -*-
from __future__ import absolute_import

__version__ = '0.0.6'

from .formatters import format_as_html, format_html_styles, format_as_text
from .explain import explain_weights, explain_prediction
from .sklearn import explain_weights_sklearn, explain_prediction_sklearn
try:
    from .lightning import (
        explain_prediction_lightning,
        explain_weights_lightning
    )
except ImportError as e:
    # lightning is not available
    pass
