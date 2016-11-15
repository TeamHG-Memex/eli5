import inspect
import six

import attr
import numpy as np

from eli5.formatters.features import FormattedFeatureName


def attrs(class_):
    """ Like attr.s with slots=True,
    but with attributes extracted from __init__ method signature.
    slots=True ensures that signature matches what really happens
    (we can't define different attributes on self).
    It is useful if we still want __init__ for proper type-checking and
    do not want to repeat attribute definitions in the class body.
    """
    attrs_kwargs = {}
    for method in ['repr', 'cmp', 'hash']:
        if '__{}__'.format(method) in class_.__dict__:
            # Allow to redefine a special method (or else attr.s will do it)
            attrs_kwargs[method] = False
    init_args = inspect.getargspec(class_.__init__)
    defaults_shift = len(init_args.args) - len(init_args.defaults or []) - 1
    these = {}
    for idx, arg in enumerate(init_args.args[1:]):
        attrib_kwargs = {}
        if idx >= defaults_shift:
            attrib_kwargs['default'] = init_args.defaults[idx - defaults_shift]
        these[arg] = attr.ib(**attrib_kwargs)
    return attr.s(class_, these=these, init=False, slots=True, **attrs_kwargs)


def numpy_to_python(obj):
    """ Convert an nested dict/list/tuple that might contain numpy objects
    to their python equivalents. Return converted object.
    """
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [numpy_to_python(x) for x in obj]
    elif isinstance(obj, FormattedFeatureName):
        return obj.value
    elif isinstance(obj, np.str_):
        return six.text_type(obj)
    elif hasattr(obj, 'dtype') and np.isscalar(obj):
        if np.issubdtype(obj, float):
            return float(obj)
        elif np.issubdtype(obj, int):
            return int(obj)
        elif np.issubdtype(obj, bool):
            return bool(obj)
    return obj
