# -*- coding: utf-8 -*-
from __future__ import absolute_import
import re

import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression

from eli5.sklearn import explain_weights_sklearn, explain_prediction_sklearn
from eli5.formatters import fields
from eli5.formatters.text import _SPACE
from .utils import format_as_all, get_all_features


def test_show_fields(newsgroups_train):
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()
    clf = LogisticRegression(C=0.1)

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    expl = explain_weights_sklearn(clf, vec)
    text, html = format_as_all(expl, clf)

    assert 'Caveats' in text
    assert 'Caveats' in html
    assert '<BIAS>' in text
    assert 'BIAS' in html

    text, html = format_as_all(expl, clf, show=fields.WEIGHTS)

    assert 'Caveats' not in text
    assert 'Caveats' not in html
    assert '<BIAS>' in text
    assert 'BIAS' in html

    text, html = format_as_all(expl, clf, show=fields.INFO)
    assert 'Caveats' in text
    assert 'Caveats' in html
    assert '<BIAS>' not in text
    assert 'BIAS' not in html


def test_formatter_order(boston_train):
    reg = LinearRegression()
    X, y, feature_names = boston_train
    reg.fit(X, y)
    res = explain_prediction_sklearn(reg, X[0], top=(3, 3))
    expl_text, expl_html = format_as_all(res, reg)
    neg_weights = get_all_features(
        res.targets[0].feature_weights.neg, with_weights=True)
    assert neg_weights['x10'] < neg_weights['x12']
    for expl in [expl_text, expl_html]:
        assert expl.find('x10') > expl.find('x12')


@pytest.mark.parametrize(
    ['hl_spaces', 'add_invisible_spaces'],
    [[hl, add] for hl in [None, True, False] for add in [True, False]])
def test_highlight_spaces(boston_train, hl_spaces, add_invisible_spaces):
    reg = LinearRegression()
    X, y, feature_names = boston_train
    # last is left unmodified to check that we are doing "any", not "all"
    modified_feature_names = \
        [('{} ' if add_invisible_spaces else 'A {}').format(name)
         for name in feature_names[:-1]] + [feature_names[-1]]
    reg.fit(X, y)
    res = explain_weights_sklearn(
        reg, feature_names=modified_feature_names, top=len(feature_names) + 1)
    expls = format_as_all(res, reg, highlight_spaces=hl_spaces)
    for is_html, expl in enumerate(expls):
        print(expl)
        if hl_spaces == False or (hl_spaces is None and not add_invisible_spaces):
            for f in modified_feature_names:
                assert f in expl
        else:
            if not add_invisible_spaces:
                assert hl_spaces
            for f in modified_feature_names[:-1]:
                assert f not in expl
            for f in feature_names[:-1]:
                if is_html:
                    if add_invisible_spaces:
                        assert re.search(
                            f + '<span.*title="A space symbol"', expl)
                    else:
                        assert re.search(
                            'A<span.*title="A space symbol".*>' + f, expl)
                else:
                    if add_invisible_spaces:
                        assert (f + _SPACE) in expl
                    else:
                        assert ('A' + _SPACE + f) in expl


def test_show_feature_values():
    clf = LinearRegression()
    clf.fit(np.array([[1, 0.1], [0, 0]]), np.array([0, 1]))
    res = explain_weights_sklearn(clf)
    for expl in format_as_all(res, clf, show_feature_values=True):
        assert 'Value' not in expl
        assert 'Weight' in expl
        assert 'x0' in expl
    res = explain_prediction_sklearn(clf, np.array([1.52, 0.5]))
    for expl in format_as_all(res, clf, show_feature_values=True):
        assert 'Contribution' in expl
        assert 'Value' in expl
        assert 'x0' in expl
        assert '1.52' in expl
    for expl in format_as_all(res, clf, show_feature_values=False):
        assert 'Contribution' in expl
        assert 'Value' not in expl
        assert 'x0' in expl
        assert '1.52' not in expl
