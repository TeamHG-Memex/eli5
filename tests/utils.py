# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import inspect
import json
from pprint import pprint

from hypothesis.strategies import integers
from hypothesis.extra.numpy import arrays

from eli5.formatters import format_as_text, format_as_html, format_as_dict
from eli5.formatters.html import html_escape
from eli5.formatters.text import format_signed


def rnd_len_arrays(dtype, min_len=0, max_len=3, elements=None):
    """ Generate numpy arrays of random length """
    lengths = integers(min_value=min_len, max_value=max_len)
    return lengths.flatmap(lambda n: arrays(dtype, n, elements=elements))


def format_as_all(res, clf, **kwargs):
    """ Format explanation as text and html, check JSON-encoding,
    print text explanation, save html, return text and html.
    """
    expl_dict = format_as_dict(res)
    pprint(expl_dict)
    json.dumps(expl_dict)  # check that it can be serialized to JSON
    expl_text = format_as_text(res, **kwargs)
    expl_html = format_as_html(res, **kwargs)
    print(expl_text)
    write_html(clf, expl_html, expl_text)
    return expl_text, expl_html


def strip_blanks(html):
    """ Remove whitespace and line breaks from html.
    """
    return html.replace(' ', '').replace('\n', '')


def write_html(clf, html, text, postfix=''):
    """ Write to html file in .html directory. Filename is generated from calling
    function name and module, and clf class name.
    This is useful to check and debug format_as_html function.
    """
    caller = inspect.stack()[2]
    try:
        test_name, test_file = caller.function, caller.filename
    except AttributeError:
        test_name, test_file = caller[3], caller[1]
    test_file = os.path.basename(test_file).rsplit('.', 1)[0]
    filename = '{}_{}_{}{}.html'.format(
        test_file, test_name, clf.__class__.__name__, postfix)
    dirname = '.html'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    path = os.path.join(dirname, filename)
    with open(path, 'wb') as f:
        f.write(u'Text:<pre>{text}</pre>End of text<hr/>\n{html}'
                .format(text=html_escape(text), html=html)
                .encode('utf8'))
    print('html written to {}'.format(path))


def get_all_features(feature_weights, with_weights=False):
    """ Collect a dict of all features and their weights.
    """
    features = {}
    for fw in feature_weights:
        if isinstance(fw.feature, list):
            features.update((f['name'], fw.weight) for f in fw.feature)
        else:
            features[fw.feature] = fw.weight
    return features if with_weights else set(features)


def get_names_coefs(feature_weights):
    return [(format_signed(fw.feature[0]) if isinstance(fw.feature, list)
             else fw.feature,
             fw.weight)
            for fw in feature_weights]
