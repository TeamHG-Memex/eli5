import inspect

import attr  # type: ignore

try:
    from functools import singledispatch  # type: ignore
except ImportError:
    from singledispatch import singledispatch  # type: ignore


def attrs(class_):
    """ Like attr.s with slots=True,
    but with attributes extracted from __init__ method signature.
    slots=True ensures that signature matches what really happens
    (we can't define different attributes on self).
    It is useful if we still want __init__ for proper type-checking and
    do not want to repeat attribute definitions in the class body.
    """
    attrs_kwargs = {}
    for method, kw_name in [
            ('__repr__', 'repr'),
            ('__eq__', 'cmp'),
            ('__hash__', 'hash'),
            ]:
        if method in class_.__dict__:
            # Allow to redefine a special method (or else attr.s will do it)
            attrs_kwargs[kw_name] = False
    init_args = inspect.getargspec(class_.__init__)
    defaults_shift = len(init_args.args) - len(init_args.defaults or []) - 1
    these = {}
    for idx, arg in enumerate(init_args.args[1:]):
        attrib_kwargs = {}
        if idx >= defaults_shift:
            attrib_kwargs['default'] = init_args.defaults[idx - defaults_shift]
        these[arg] = attr.ib(**attrib_kwargs)
    return attr.s(class_, these=these, init=False, slots=True, **attrs_kwargs)  # type: ignore
