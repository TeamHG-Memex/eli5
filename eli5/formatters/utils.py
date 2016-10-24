import re


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
