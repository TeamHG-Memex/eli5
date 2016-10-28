# -*- coding: utf-8 -*-
from __future__ import absolute_import
from pprint import pprint

import pytest

from eli5.sklearn import InvertableHashingVectorizer
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from eli5.sklearn import explain_prediction
from .utils import format_as_all, get_all_features, get_names_coefs


def check_explain_linear_binary(res, clf):
    expl_text, expl_html = format_as_all(res, clf)
    assert len(res['classes']) == 1
    e = res['classes'][0]
    assert e['class'] == 'comp.graphics'
    neg = get_all_features(e['feature_weights']['neg'])
    assert 'objective' in neg
    for expl in [expl_text, expl_html]:
        assert 'comp.graphics' in expl
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
        clf, docs[0], vec, target_names=target_names, top=20)
    res = get_res()
    check_explain_linear_binary(res, clf)
    assert res == get_res()
    res_vectorized = explain_prediction(
        clf, vec.transform([docs[0]])[0], vec, target_names=target_names,
        top=20, vectorized=True)
    if isinstance(vec, HashingVectorizer):
        # InvertableHashingVectorizer must be passed with vectorized=True
        neg_vectorized = get_all_features(
            res_vectorized['classes'][0]['feature_weights']['neg'])
        assert all(name.startswith('x') for name in neg_vectorized)
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
        clf, docs[0], ivec, target_names=target_names, top=20, **kwargs)
    res = get_res()
    check_explain_linear_binary(res, clf)
    assert res == get_res()
    res_vectorized = explain_prediction(
        clf, vec.transform([docs[0]])[0], ivec, target_names=target_names,
        top=20, vectorized=True)
    pprint(res_vectorized)
    assert res_vectorized == _without_weighted_spans(res)

    assert res == get_res(
        feature_names=ivec.get_feature_names(always_signed=False))


def _without_weighted_spans(res):
    res = dict(res)
    res['classes'] = [{k: v for k, v in d.items() if k != 'weighted_spans'}
                       for d in res['classes']]
    return res


def test_explain_linear_dense():
    clf = LogisticRegression()
    data = [{'day': 'mon', 'moon': 'full'},
            {'day': 'tue', 'moon': 'rising'},
            {'day': 'tue', 'moon': 'rising'},
            {'day': 'mon', 'moon': 'rising'}]
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(data)
    clf.fit(X, [0, 1, 1, 0])
    test_day = {'day': 'tue', 'moon': 'full'}
    target_names = ['sunny', 'shady']
    res1 = explain_prediction(clf, test_day, vec, target_names=target_names)
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
    res = explain_prediction(clf, 'hello, world', vec)
    assert 'Error' in res['description']


def test_explain_regression_hashing_vectorizer(newsgroups_train_binary):
    docs, y, target_names = newsgroups_train_binary
    vec = HashingVectorizer(norm=None)
    clf = LinearRegression()
    clf.fit(vec.fit_transform(docs), y)

    # Setting large "top" in order to compare it with CountVectorizer below
    # (due to small differences in the coefficients they might have cutoffs
    # at different points).
    res = explain_prediction(
        clf, docs[0], vec, target_names=[target_names[1]], top=1000)
    expl, _ = format_as_all(res, clf)
    assert len(res['targets']) == 1
    e = res['targets'][0]
    assert e['target'] == 'comp.graphics'
    neg = get_all_features(e['feature_weights']['neg'])
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
        count_clf, docs[0], count_vec, target_names=[target_names[1]], top=1000)
    pprint(count_res)
    count_expl, _ = format_as_all(count_res, count_clf)
    print(count_expl)

    for key in ['pos', 'neg']:
        values, count_values = [
            sorted(get_names_coefs(r['targets'][0]['feature_weights'][key]))
            for r in [res, count_res]]
        assert len(values) == len(count_values)
        for (name, coef), (count_name, count_coef) in zip(values, count_values):
            assert name == count_name
            assert abs(coef - count_coef) < 0.05
