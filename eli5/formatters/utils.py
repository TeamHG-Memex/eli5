import re
from typing import Union

from eli5.base import Explanation


def replace_spaces(s, replacer):
    """
    >>> replace_spaces('ab', lambda n, l: '_' * n)
    'ab'
    >>> replace_spaces('a b', lambda n, l: '_' * n)
    'a_b'
    >>> replace_spaces(' ab', lambda n, l: '_' * n)
    '_ab'
    >>> replace_spaces('  a b ', lambda n, s: s * n)
    'leftleftacenterbright'
    >>> replace_spaces(' a b  ', lambda n, _: '0 0' * n)
    '0 0a0 0b0 00 0'
    """
    def replace(m):
        if m.start() == 0:
            side = 'left'
        elif m.end() == len(s):
            side = 'right'
        else:
            side = 'center'
        return replacer(len(m.group()), side)

    return re.sub(r'[ ]+', replace, s)


def format_signed(feature, formatter=None, **kwargs):
    """
    Format unhashed feature with sign.

    >>> format_signed({'name': 'foo', 'sign': 1})
    'foo'
    >>> format_signed({'name': 'foo', 'sign': -1})
    '(-)foo'
    >>> format_signed({'name': ' foo', 'sign': -1}, lambda x: '"{}"'.format(x))
    '(-)" foo"'
    """
    txt = '' if feature['sign'] > 0 else '(-)'
    name = feature['name']
    if formatter is not None:
        name = formatter(name, **kwargs)
    return '{}{}'.format(txt, name)


def should_highlight_spaces(explanation):
    # type: (Explanation) -> bool
    hl_spaces = explanation.highlight_spaces
    if explanation.feature_importances:
        hl_spaces = hl_spaces or any(
            _has_invisible_spaces(fw.feature)
            for fw in explanation.feature_importances)
    if explanation.targets:
        hl_spaces = hl_spaces or any(
            _has_invisible_spaces(fw.feature)
            for target in explanation.targets
            for weights in [target.feature_weights.pos, target.feature_weights.neg]
            for fw in weights)
    return hl_spaces


def _has_invisible_spaces(name):
    # type: (Union[str, List[Dict]]) -> bool
    if isinstance(name, list):
        return any(_has_invisible_spaces(n['name']) for n in name)
    return name.startswith(' ') or name.endswith(' ')


def max_or_0(it):
    lst = list(it)
    return max(lst) if lst else 0
