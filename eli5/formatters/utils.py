from __future__ import absolute_import
from itertools import chain
import re
import six
from typing import Any, Union, List, Dict, Callable, Match, Optional

import numpy as np

from eli5.base import Explanation
from .features import FormattedFeatureName


def replace_spaces(s, replacer):
    # type: (str, Callable[[int, str], str]) -> str
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
        # type: (Match[str]) -> str
        if m.start() == 0:
            side = 'left'
        elif m.end() == len(s):
            side = 'right'
        else:
            side = 'center'
        return replacer(len(m.group()), side)

    return re.sub(r'[ ]+', replace, s)


def format_signed(feature,  # type: Dict[str, Any]
                  formatter=None,  # type: Callable[..., str]
                  **kwargs
                  ):
    # type: (...) -> str
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
    name = feature['name']  # type: str
    if formatter is not None:
        name = formatter(name, **kwargs)
    return '{}{}'.format(txt, name)


def should_highlight_spaces(explanation):
    # type: (Explanation) -> bool
    hl_spaces = bool(explanation.highlight_spaces)
    if explanation.feature_importances:
        hl_spaces = hl_spaces or any(
            _has_invisible_spaces(fw.feature)
            for fw in explanation.feature_importances.importances)
    if explanation.targets:
        hl_spaces = hl_spaces or any(
            _has_invisible_spaces(fw.feature)
            for target in explanation.targets if target.feature_weights is not None
            for weights in [target.feature_weights.pos, target.feature_weights.neg]
            for fw in weights)
    return hl_spaces


def _has_invisible_spaces(name):
    # type: (Union[str, List[Dict], FormattedFeatureName]) -> bool
    if isinstance(name, FormattedFeatureName):
        return False
    elif isinstance(name, list):
        return any(_has_invisible_spaces(n['name']) for n in name)
    else:
        return name.startswith(' ') or name.endswith(' ')


def has_any_values_for_weights(explanation):
    # type: (Explanation) -> bool
    if explanation.targets:
        return any(fw.value is not None
                   for t in explanation.targets
            if t.feature_weights is not None 
            for fw in chain(
            t.feature_weights.pos, t.feature_weights.neg))
    else:
        return False


def tabulate(data,  # type: List[List[Any]]
             header=None,  # type: Optional[List[Any]]
             col_align=None,  # type: Union[str, List[str]]
             ):
    # type: (...) -> List[str]
    """ Format data as a table without any fancy features.
    col_align: l/r/c or a list/string of l/r/c. l = left, r = right, c = center
    Return a list of strings (lines of the table).
    """
    if not data and not header:
        return []
    if data:
        n_cols = len(data[0])
    else:
        assert header is not None
        n_cols = len(header)
    if not all(len(row) == n_cols for row in data):
        raise ValueError('data is not rectangular')

    if col_align is None:
        col_align = ['l'] * n_cols
    elif isinstance(col_align, six.string_types) and len(col_align) == 1:
        col_align = [col_align] * n_cols
    else:
        col_align = list(col_align)
        if len(col_align) != n_cols:
            raise ValueError('col_align length does not match number of columns')

    if header and len(header) != n_cols:
        raise ValueError('header length does not match number of columns')

    if header:
        data = [header] + data
    data = [[six.text_type(x) for x in row] for row in data]
    col_width = [max(len(row[col_i]) for row in data) for col_i in range(n_cols)]
    if header:
        data.insert(1, ['-' * width for width in col_width])

    line_tpl = u'  '.join(
        u'{:%s%s}' % ({'l': '', 'r': '>', 'c': '^'}[align], width)
        for align, width in zip(col_align, col_width))
    return [line_tpl.format(*row) for row in data]


def format_weight(value):
    # type: (float) -> str
    return '{:+.3f}'.format(value)


def format_value(value):
    # type: (Optional[float]) -> str
    if value is None:
        return ''
    elif np.isnan(value):
        return 'Missing'
    else:
        return '{:.3f}'.format(value)
