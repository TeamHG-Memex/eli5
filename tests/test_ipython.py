# -*- coding: utf-8 -*-
import pytest
pytest.importorskip('IPython')

import numpy as np
from sklearn.linear_model import LogisticRegression
from IPython.display import HTML

import eli5
from .utils import write_html


def test_show_weights():
    clf = LogisticRegression()
    X = [[0, 0], [1, 1], [0, 1]]
    y = ['a', 'b', 'a']
    clf.fit(X, y)

    html = eli5.show_weights(clf)
    # write_html(clf, html.data, '')
    assert isinstance(html, HTML)
    assert 'y=b' in html.data
    assert 'Explained as' not in html.data

    # explain_weights arguments are supported
    html = eli5.show_weights(clf, target_names=['A', 'B'])
    assert 'y=B' in html.data

    # format_as_html arguments are supported
    html = eli5.show_weights(clf, show=['method'])
    assert 'y=b' not in html.data
    assert 'Explained as' in html.data


def test_show_prediction():
    clf = LogisticRegression(C=100)
    X = [[0, 0], [1, 1], [0, 1]]
    y = ['a', 'b', 'a']
    clf.fit(X, y)

    doc = np.array([0, 1])

    html = eli5.show_prediction(clf, doc)
    write_html(clf, html.data, '')
    assert isinstance(html, HTML)
    assert 'y=a' in html.data
    assert 'BIAS' in html.data
    assert 'x1' in html.data

    # explain_prediction arguments are supported
    html = eli5.show_prediction(clf, doc, feature_names=['foo', 'bar'])
    write_html(clf, html.data, '')
    assert 'x1' not in html.data
    assert 'bar' in html.data

    # format_as_html arguments are supported
    html = eli5.show_prediction(clf, doc, show=['method'])
    write_html(clf, html.data, '')
    assert 'y=a' not in html.data
    assert 'BIAS' not in html.data
    assert 'Explained as' in html.data

    # top target is used
    html = eli5.show_prediction(clf, np.array([1, 1]))
    write_html(clf, html.data, '')
    assert 'y=b' in html.data
    assert 'BIAS' in html.data
    assert 'x1' in html.data

