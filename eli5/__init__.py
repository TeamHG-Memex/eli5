# -*- coding: utf-8 -*-
from __future__ import absolute_import

__version__ = '0.6.2'

from .formatters import (
    format_as_html,
    format_html_styles,
    format_as_text,
    format_as_dict
)
from .explain import explain_weights, explain_prediction
from .sklearn import explain_weights_sklearn, explain_prediction_sklearn
from .transform import transform_feature_names


try:
    from .ipython import show_weights, show_prediction
except ImportError:
    pass  # IPython is not installed


try:
    from .lightning import (
        explain_prediction_lightning,
        explain_weights_lightning
    )
except ImportError as e:
    # lightning is not available
    pass


try:
    from .sklearn_crfsuite import (
        explain_weights_sklearn_crfsuite
    )
except ImportError as e:
    # sklearn-crfsuite is not available
    pass


try:
    from .xgboost import (
        explain_weights_xgboost,
        explain_prediction_xgboost
    )
except ImportError:
    # xgboost is not available
    pass
except Exception as e:
    if e.__class__.__name__ == 'XGBoostLibraryNotFound':
        # improperly installed xgboost
        pass
    else:
        raise


try:
    from .lightgbm import (
        explain_weights_lightgbm,
        explain_prediction_lightgbm
    )
except ImportError:
    # lightgbm is not available
    pass