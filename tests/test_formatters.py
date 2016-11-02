# -*- coding: utf-8 -*-
from __future__ import absolute_import

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression

from eli5.sklearn import explain_weights_sklearn, explain_prediction_sklearn
from eli5.formatters import fields
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
