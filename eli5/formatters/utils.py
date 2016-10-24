import re


def replace_starting_trailing_spaces(s, replacer):
    """
    >>> replace_starting_trailing_spaces('ab', lambda n, l: '_' * n)
    'ab'
    >>> replace_starting_trailing_spaces('a b', lambda n, l: '_' * n)
    'a b'
    >>> replace_starting_trailing_spaces(' ab', lambda n, l: '_' * n)
    '_ab'
    >>> replace_starting_trailing_spaces('  a b   ', lambda n, l: '_' * (n + l))
    '___a b___'
    """
    s = re.sub('^[ ]+', lambda m: replacer(len(m.group()), True), s)
    s = re.sub('[ ]+$', lambda m: replacer(len(m.group()), False), s)
    return s


def format_signed(feature, formatter=None):
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
        name = formatter(name)
    return '{}{}'.format(txt, name)
