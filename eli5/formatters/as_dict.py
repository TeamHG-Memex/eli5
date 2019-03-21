import six

import attr   # type: ignore
import numpy as np   # type: ignore

from .features import FormattedFeatureName


def format_as_dict(explanation):
    """ Return a dictionary representing the explanation that can be JSON-encoded.
    It accepts parts of explanation (for example feature weights) as well.
    """
    return _numpy_to_python(attr.asdict(explanation))


_numpy_string_types = (np.string_, np.unicode_) if six.PY2 else np.str_


def _numpy_to_python(obj):
    """ Convert an nested dict/list/tuple that might contain numpy objects
    to their python equivalents. Return converted object.
    """
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [_numpy_to_python(x) for x in obj]
    elif isinstance(obj, FormattedFeatureName):
        return obj.value
    elif isinstance(obj, _numpy_string_types):
        return six.text_type(obj)
    elif hasattr(obj, 'dtype') and np.isscalar(obj):
        if np.issubdtype(obj, np.floating):
            return float(obj)
        elif np.issubdtype(obj, np.integer):
            return int(obj)
        elif np.issubdtype(obj, np.bool_):
            return bool(obj)
    return obj
