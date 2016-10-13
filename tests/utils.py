# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import inspect

from hypothesis.strategies import integers
from hypothesis.extra.numpy import arrays


def rnd_len_arrays(dtype, min_len=0, max_len=3, elements=None):
    """ Generate numpy arrays of random length """
    lengths = integers(min_value=min_len, max_value=max_len)
    return lengths.flatmap(lambda n: arrays(dtype, n, elements=elements))


def write_html(clf, html):
    """ Write to html file in .html directory. Filename is generated from calling
    function name and module, and clf class name.
    This is useful to check and debug format_as_html function.
    """
    caller = inspect.stack()[1]
    try:
        test_name, test_file = caller.function, caller.filename
    except AttributeError:
        test_name, test_file = caller[3], caller[1]
    test_file = os.path.basename(test_file).rsplit('.', 1)[0]
    filename = '{}_{}_{}.html'.format(
        test_file, test_name, clf.__class__.__name__)
    dirname = '.html'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    path = os.path.join(dirname, filename)
    with open(path, 'wb') as f:
        f.write(html.encode('utf8'))
    print('html written to {}'.format(path))
