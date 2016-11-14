# -*- coding: utf-8 -*-
from __future__ import absolute_import

import pytest
from sklearn_crfsuite import CRF

from eli5 import explain_weights
from .utils import format_as_all


@pytest.fixture()
def xseq():
    return [
        {'walk': 1, 'shop': 0.5},
        {'walk': 1},
        {'walk': 1, 'clean': 0.5},
        {u'shop': 0.5, u'clean': 0.5},
        {'walk': 0.5, 'clean': 1},
        {'clean': 1, u'shop': 0.1},
        {'walk': 1, 'shop': 0.5},
        {},
        {'clean': 1},
        {u'солнце': u'не светит'.encode('utf8'), 'clean': 1},
    ]


@pytest.fixture()
def yseq():
    return ['sunny', 'sunny', u'sunny', 'rainy', 'rainy', 'rainy',
            'sunny', 'sunny', 'rainy', 'rainy']


def test_sklearn_crfsuite(xseq, yseq):
    crf = CRF(c1=0.0, c2=0.1, max_iterations=50)
    crf.fit([xseq], [yseq])

    expl = explain_weights(crf)
    text, html = format_as_all(expl, crf)

    assert "y='sunny' top features" in text
    assert "y='rainy' top features" in text
    assert "Transition features" in text
    assert "sunny    0.696   -0.130" in text
    assert u'+0.124  солнце:не светит' in text

    html_nospaces = html.replace(' ', '').replace("\n", '')
    assert u'солнце:не светит' in html
    assert '<th>sunny</th><th>rainy</th>' in html_nospaces


def test_sklearn_crfsuite_feature_re(xseq, yseq):
    crf = CRF(c1=0.0, c2=0.1, max_iterations=50)
    crf.fit([xseq], [yseq])

    expl = explain_weights(crf, feature_re=u'(солн|clean)')
    for expl in format_as_all(expl, crf):
        assert u'солн' in expl
        assert u'clean' in expl
        assert 'walk' not in expl


def test_sklearn_targets(xseq, yseq):
    crf = CRF(c1=0.0, c2=0.1, max_iterations=50)
    crf.fit([xseq], [yseq])
    res = explain_weights(crf,
                          target_names={'sunny': u'☀'},
                          target_order=['rainy', 'sunny'])
    for expl in format_as_all(res, crf):
        assert u'☀' in expl
        assert expl.index('rainy') < expl.index(u'☀')

