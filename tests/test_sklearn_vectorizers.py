# -*- coding: utf-8 -*-
from __future__ import absolute_import
from pprint import pprint

import attr
import pytest

from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from eli5 import explain_prediction, explain_weights
from eli5.formatters import format_as_html
from eli5.sklearn import invert_hashing_and_fit, InvertableHashingVectorizer
from .utils import format_as_all, get_all_features, get_names_coefs, write_html


def check_explain_linear_binary(res, clf, target='alt.atheism'):
    expl_text, expl_html = format_as_all(res, clf)
    assert len(res.targets) == 1
    e = res.targets[0]
    assert e.target == target
    pos = get_all_features(e.feature_weights.pos)
    assert 'objective' in pos
    for expl in [expl_text, expl_html]:
        assert target in expl
        assert 'objective' in expl


@pytest.mark.parametrize(['vec'], [
    [CountVectorizer()],
    [HashingVectorizer()],
])
def test_explain_linear_binary(vec, newsgroups_train_binary):
    docs, y, target_names = newsgroups_train_binary
    clf = LogisticRegression(random_state=42)
    X = vec.fit_transform(docs)
    clf.fit(X, y)

    get_res = lambda: explain_prediction(
        clf, docs[0], vec=vec, target_names=target_names, top=20)
    res = get_res()
    check_explain_linear_binary(res, clf)
    assert res == get_res()
    res_vectorized = explain_prediction(
        clf, vec.transform([docs[0]])[0], vec=vec, target_names=target_names,
        top=20, vectorized=True)
    if isinstance(vec, HashingVectorizer):
        # InvertableHashingVectorizer must be passed with vectorized=True
        pos_weights = res_vectorized.targets[0].feature_weights.pos
        pos_vectorized = get_all_features(pos_weights)
        assert all(name.startswith('x') for name in pos_vectorized)
    else:
        assert res_vectorized == _without_weighted_spans(res)


def test_explain_hashing_vectorizer(newsgroups_train_binary):
    # test that we can pass InvertableHashingVectorizer explicitly
    vec = HashingVectorizer(n_features=1000)
    ivec = InvertableHashingVectorizer(vec)
    clf = LogisticRegression(random_state=42)
    docs, y, target_names = newsgroups_train_binary
    ivec.fit([docs[0]])
    X = vec.fit_transform(docs)
    clf.fit(X, y)

    get_res = lambda **kwargs: explain_prediction(
        clf, docs[0], vec=ivec, target_names=target_names, top=20, **kwargs)
    res = get_res()
    check_explain_linear_binary(res, clf)
    assert res == get_res()
    res_vectorized = explain_prediction(
        clf, vec.transform([docs[0]])[0], vec=ivec, target_names=target_names,
        top=20, vectorized=True)
    pprint(res_vectorized)
    assert res_vectorized == _without_weighted_spans(res)

    assert res == get_res(
        feature_names=ivec.get_feature_names(always_signed=False))


def _without_weighted_spans(res):
    return attr.assoc(res, targets=[
        attr.assoc(target, weighted_spans=None) for target in res.targets])


def test_explain_linear_dense():
    clf = LogisticRegression(random_state=42)
    data = [{'day': 'mon', 'moon': 'full'},
            {'day': 'tue', 'moon': 'rising'},
            {'day': 'tue', 'moon': 'rising'},
            {'day': 'mon', 'moon': 'rising'}]
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(data)
    clf.fit(X, [0, 1, 1, 0])
    test_day = {'day': 'tue', 'moon': 'full'}
    target_names = ['sunny', 'shady']
    res1 = explain_prediction(clf, test_day, vec=vec, target_names=target_names)
    expl_text, expl_html = format_as_all(res1, clf)
    assert 'day=tue' in expl_text
    assert 'day=tue' in expl_html
    [test_day_vec] = vec.transform(test_day)
    res2 = explain_prediction(
        clf, test_day_vec, target_names=target_names,
        vectorized=True, feature_names=vec.get_feature_names())
    assert res1 == res2


def test_unsupported():
    vec = CountVectorizer()
    clf = BaseEstimator()
    res = explain_prediction(clf, 'hello, world', vec=vec)
    assert 'BaseEstimator' in res.error
    for expl in format_as_all(res, clf):
        assert 'Error' in expl
        assert 'BaseEstimator' in expl


def test_explain_regression_hashing_vectorizer(newsgroups_train_binary):
    docs, y, target_names = newsgroups_train_binary
    vec = HashingVectorizer(norm=None)
    clf = LinearRegression()
    clf.fit(vec.fit_transform(docs), y)

    # Setting large "top" in order to compare it with CountVectorizer below
    # (due to small differences in the coefficients they might have cutoffs
    # at different points).
    res = explain_prediction(
        clf, docs[0], vec=vec, target_names=[target_names[1]], top=1000)
    expl, _ = format_as_all(res, clf)
    assert len(res.targets) == 1
    e = res.targets[0]
    assert e.target == 'comp.graphics'
    neg = get_all_features(e.feature_weights.neg)
    assert 'objective' in neg
    assert 'that' in neg
    assert 'comp.graphics' in expl
    assert 'objective' in expl
    assert 'that' in expl

    # HashingVectorizer with norm=None is "the same" as CountVectorizer,
    # so we can compare it and check that explanation is almost the same.
    count_vec = CountVectorizer()
    count_clf = LinearRegression()
    count_clf.fit(count_vec.fit_transform(docs), y)
    count_res = explain_prediction(
        count_clf, docs[0], vec=count_vec, target_names=[target_names[1]],
        top=1000)
    pprint(count_res)
    count_expl, _ = format_as_all(count_res, count_clf)
    print(count_expl)

    for key in ['pos', 'neg']:
        values, count_values = [
            sorted(get_names_coefs(getattr(r.targets[0].feature_weights, key)))
            for r in [res, count_res]]
        assert len(values) == len(count_values)
        for (name, coef), (count_name, count_coef) in zip(values, count_values):
            assert name == count_name
            assert abs(coef - count_coef) < 0.05


@pytest.mark.parametrize(['vec_cls'], [
    [CountVectorizer],
    [HashingVectorizer],
])
def test_explain_feature_union(vec_cls):
    data = [
        {'url': 'http://a.com/blog',
         'text': 'security research'},
        {'url': 'http://a.com',
         'text': 'security research'},
        {'url': 'http://b.com/blog',
         'text': 'health study'},
        {'url': 'http://b.com',
         'text': 'health research'},
        {'url': 'http://c.com/blog',
         'text': 'security'},
        ]
    ys = [1, 0, 0, 0, 1]
    url_vec = vec_cls(
        preprocessor=lambda x: x['url'], analyzer='char', ngram_range=(3, 3))
    text_vec = vec_cls(preprocessor=lambda x: x['text'])
    vec = FeatureUnion([('url', url_vec), ('text', text_vec)])
    xs = vec.fit_transform(data)
    clf = LogisticRegression(random_state=42)
    clf.fit(xs, ys)

    ivec = invert_hashing_and_fit(vec, data)
    weights_res = explain_weights(clf, ivec)
    html_expl = format_as_html(weights_res)
    write_html(clf, html_expl, '', postfix='{}_weights'.format(vec_cls.__name__))
    assert 'text__security' in html_expl
    assert 'url__log' in html_expl
    assert 'BIAS' in html_expl

    pred_res = explain_prediction(clf, data[0], vec)
    html_expl = format_as_html(pred_res, force_weights=False)
    write_html(clf, html_expl, '', postfix=vec_cls.__name__)
    assert 'text: Highlighted in text (sum)' in html_expl
    assert 'url: Highlighted in text (sum)' in html_expl
    assert '<b>url:</b> <span' in html_expl
    assert '<b>text:</b> <span' in html_expl
    assert 'BIAS' in html_expl


@pytest.mark.parametrize(['vec_cls'], [
    [CountVectorizer],
    [HashingVectorizer],
])
def test_explain_feature_union_with_nontext(vec_cls):
    data = [
        {'score': 1,
         'text': 'security research'},
        {'score': 0.1,
         'text': 'security research'},
        {'score': 0.5,
         'text': 'health study'},
        {'score': 0.5,
         'text': 'health research'},
        {'score': 0.1,
         'text': 'security'},
    ]
    ys = [1, 0, 0, 0, 1]
    score_vec = DictVectorizer()
    text_vec = vec_cls(preprocessor=lambda x: x['text'])
    vec = FeatureUnion([('score', score_vec), ('text', text_vec)])
    xs = vec.fit_transform(data)
    clf = LogisticRegression(random_state=42)
    clf.fit(xs, ys)

    ivec = invert_hashing_and_fit(vec, data)
    weights_res = explain_weights(clf, ivec)
    html_expl = format_as_html(weights_res)
    write_html(clf, html_expl, '', postfix='{}_weights'.format(vec_cls.__name__))
    assert 'score__score' in html_expl
    assert 'text__security' in html_expl
    assert 'BIAS' in html_expl

    res = explain_prediction(clf, data[0], vec)
    html_expl = format_as_html(res, force_weights=False)
    write_html(clf, html_expl, '', postfix=vec_cls.__name__)
    assert 'text: Highlighted in text (sum)' in html_expl
    assert '<b>text:</b> <span' in html_expl
    assert 'BIAS' in html_expl
    assert 'score__score' in html_expl
