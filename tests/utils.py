# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import inspect
import json
from pprint import pprint

from hypothesis.strategies import integers
from hypothesis.extra.numpy import arrays
import numpy as np

from eli5.base import Explanation
from eli5.formatters import format_as_text, format_as_html, format_as_dict
from eli5.formatters.html import html_escape
from eli5.formatters.text import format_signed
from eli5.sklearn.utils import sklearn_version


SGD_KWARGS = {'random_state': 42}
if sklearn_version() >= '0.19':
    SGD_KWARGS['tol'] = 1e-3


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
    write_html(clf, expl_html, expl_text, caller_depth=2)
    return expl_text, expl_html


def strip_blanks(html):
    """ Remove whitespace and line breaks from html.
    """
    return html.replace(' ', '').replace('\n', '')


def write_html(clf, html, text, postfix='', caller_depth=1):
    """ Write to html file in .html directory. Filename is generated from calling
    function name and module, and clf class name.
    This is useful to check and debug format_as_html function.
    """
    caller = inspect.stack()[caller_depth]
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


def check_targets_scores(explanation, atol=1e-8):
    # type: (Explanation, float) -> None
    """ Check that feature weights sum to target score or proba,
    if both proba and score are present they match,
    and that there are no "remaining" features.
    """
    targets = explanation.targets
    for target in targets:
        weights = target.feature_weights
        # else the check is invalid
        assert weights.neg_remaining == weights.pos_remaining == 0
        weights_sum = (sum(fw.weight for fw in weights.pos) +
                       sum(fw.weight for fw in weights.neg))
        expected = target.score if target.score is not None else target.proba
        assert np.isclose(abs(expected), abs(weights_sum), atol=atol), \
            (expected, weights_sum)
    if any(t.score is not None for t in targets):
        if len(targets) == 1 and targets[0].proba is not None:
            target = targets[0]
            # one target with proba => assume sigmoid
            proba = 1. / (1 + np.exp(-target.score))
            assert np.isclose(target.proba, proba, atol=atol) or \
                   np.isclose(target.proba, 1-proba, atol=atol)
        elif any(t.proba is not None for t in targets):
            # many targets with proba => assume softmax
            norm = np.sum(np.exp([t.score for t in targets]))
            for target in targets:
                assert np.isclose(np.exp(target.score) / norm, target.proba,
                                  atol=atol)
