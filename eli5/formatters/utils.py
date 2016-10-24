import re


def replace_starting_trailing_spaces(s, replace_with):
    """
    >>> replace_starting_trailing_spaces('ab', '_')
    'ab'
    >>> replace_starting_trailing_spaces('a b', '_')
    'a b'
    >>> replace_starting_trailing_spaces(' ab', '_')
    '_ab'
    >>> replace_starting_trailing_spaces('  a b   ', '_')
    '__a b___'
    """
    repl = lambda m: m.group().replace(' ', replace_with)
    s = re.sub('^[ ]+', repl, s)
    s = re.sub('[ ]+$', repl, s)
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
