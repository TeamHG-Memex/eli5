# -*- coding: utf-8 -*-
from __future__ import absolute_import

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from eli5.sklearn import explain_weights
from eli5.formatters import fields
from .utils import format_as_all


def test_show_fields(newsgroups_train):
    docs, y, target_names = newsgroups_train
    vec = TfidfVectorizer()
    clf = LogisticRegression(C=0.1)

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    expl = explain_weights(clf, vec)
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
