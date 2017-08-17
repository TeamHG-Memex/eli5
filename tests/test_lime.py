# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
import pytest

from eli5.lime import TextExplainer
from eli5.formatters import format_as_text
from .utils import format_as_all, check_targets_scores


def test_lime_explain_probabilistic(newsgroups_train):
    docs, y, target_names = newsgroups_train
    try:
        vec = HashingVectorizer(alternate_sign=False)
    except TypeError:
        # sklearn < 0.19
        vec = HashingVectorizer(non_negative=True)
    clf = MultinomialNB()

    X = vec.fit_transform(docs)
    clf.fit(X, y)
    print(clf.score(X, y))

    pipe = make_pipeline(vec, clf)
    doc = docs[0]

    te = TextExplainer(random_state=42)
    te.fit(doc, pipe.predict_proba)

    print(te.metrics_)
    assert te.metrics_['score'] > 0.7
    assert te.metrics_['mean_KL_divergence'] < 0.1

    res = te.explain_prediction(top=20, target_names=target_names)
    expl = format_as_text(res)
    print(expl)
    assert 'file' in expl


def test_lime_flat_neighbourhood(newsgroups_train):
    docs, y, target_names = newsgroups_train
    doc = docs[0]

    @_apply_to_list
    def predict_proba(doc):
        """ This function predicts non-zero probabilities only for 3 labels """
        proba_graphics = [0, 1.0, 0, 0]
        proba_other = [0.9, 0, 0.1, 0]
        return proba_graphics if 'file' in doc else proba_other

    te = TextExplainer(expand_factor=None, random_state=42)
    te.fit(doc, predict_proba)
    print(te.metrics_)
    print(te.clf_.classes_, target_names)

    res = te.explain_prediction(top=20, target_names=target_names)
    for expl in format_as_all(res, te.clf_):
        assert 'file' in expl
        assert "comp.graphics" in expl


@pytest.mark.parametrize(['token_pattern'],
                         [[None], ['.']])
def test_text_explainer_char_based(token_pattern):
    text = "Hello, world!"
    predict_proba = substring_presence_predict_proba('lo')

    te = TextExplainer(char_based=True, token_pattern=token_pattern)
    te.fit(text, predict_proba)
    print(te.metrics_)
    assert te.metrics_['score'] > 0.95
    assert te.metrics_['mean_KL_divergence'] < 0.1

    res = te.explain_prediction()
    format_as_all(res, te.clf_)
    check_targets_scores(res)
    assert res.targets[0].feature_weights.pos[0].feature == 'lo'

    # another way to look at results (not that useful for char ngrams)
    res = te.explain_weights()
    assert res.targets[0].feature_weights.pos[0].feature == 'lo'


def test_text_explainer_position_dependent():
    text = "foo bar baz egg spam bar baz egg spam ham"

    @_apply_to_list
    def predict_proba(doc):
        tokens = doc.split()
        # 'bar' is only important in the beginning of the document,
        # not in the end
        return [0, 1] if len(tokens) >= 2 and tokens[1] == 'bar' else [1, 0]

    # bag of words model is not powerful enough to explain predict_proba above
    te = TextExplainer(random_state=42, vec=CountVectorizer())
    te.fit(text, predict_proba)
    print(te.metrics_)
    assert te.metrics_['score'] < 0.9
    assert te.metrics_['mean_KL_divergence'] > 0.3

    # position_dependent=True can make it work
    te = TextExplainer(position_dependent=True, random_state=42)
    te.fit(text, predict_proba)
    print(te.metrics_)
    assert te.metrics_['score'] > 0.95
    assert te.metrics_['mean_KL_divergence'] < 0.3

    expl = te.explain_prediction()
    format_as_all(expl, te.clf_)

    # it is also possible to almost make it work using a custom vectorizer
    vec = CountVectorizer(ngram_range=(1, 2))
    te = TextExplainer(vec=vec, random_state=42)
    te.fit(text, predict_proba)
    print(te.metrics_)
    assert te.metrics_['score'] > 0.95
    assert te.metrics_['mean_KL_divergence'] < 0.3

    expl = te.explain_prediction()
    format_as_all(expl, te.clf_)

    # custom vectorizers are not supported when position_dependent is True
    with pytest.raises(ValueError):
        te = TextExplainer(position_dependent=True, vec=HashingVectorizer())


def test_text_explainer_custom_classifier():
    text = "foo-bar baz egg-spam"
    predict_proba = substring_presence_predict_proba('bar')

    # use decision tree to explain the prediction
    te = TextExplainer(clf=DecisionTreeClassifier(max_depth=2))
    te.fit(text, predict_proba)
    print(te.metrics_)
    assert te.metrics_['score'] > 0.99
    assert te.metrics_['mean_KL_divergence'] < 0.01
    expl = te.explain_prediction()
    format_as_all(expl, te.clf_)

    # with explain_weights we can get a nice tree representation
    expl = te.explain_weights()
    print(expl.decision_tree.tree)
    assert expl.decision_tree.tree.feature_name == "bar"
    format_as_all(expl, te.clf_)


def test_text_explainer_token_pattern():
    text = "foo-bar baz egg-spam"
    predict_proba = substring_presence_predict_proba('bar')

    # a different token_pattern
    te = TextExplainer(token_pattern=r'(?u)\b[-\w]+\b')
    te.fit(text, predict_proba)
    print(te.metrics_)
    assert te.metrics_['score'] > 0.95
    assert te.metrics_['mean_KL_divergence'] < 0.1
    expl = te.explain_prediction()
    format_as_all(expl, te.clf_)

    assert expl.targets[0].feature_weights.pos[0].feature == 'foo-bar'


def test_text_explainer_show_methods():
    pytest.importorskip('IPython')
    from IPython.display import HTML

    text = "Hello, world!"

    @_apply_to_list
    def predict_proba(doc):
        return [0.0, 1.0] if 'lo' in doc else [1.0, 0.0]

    te = TextExplainer()
    te.fit(text, predict_proba)

    pred_expl = te.show_prediction()
    assert isinstance(pred_expl, HTML)
    assert 'lo' in pred_expl.data

    weight_expl = te.show_weights()
    assert isinstance(weight_expl, HTML)
    assert 'lo' in weight_expl.data


def test_text_explainer_rbf_sigma():
    text = 'foo bar baz egg spam'
    predict_proba = substring_presence_predict_proba('bar')

    te1 = TextExplainer().fit(text, predict_proba)
    te2 = TextExplainer(rbf_sigma=0.1).fit(text, predict_proba)
    te3 = TextExplainer(rbf_sigma=1.0).fit(text, predict_proba)

    assert te1.similarity_.sum() < te3.similarity_.sum()
    assert te1.similarity_.sum() > te2.similarity_.sum()


def substring_presence_predict_proba(substring):
    @_apply_to_list
    def predict_proba(doc):
        return [0, 1] if substring in doc else [1, 0]
    return predict_proba


def _apply_to_list(func):
    def wrapper(docs):
        return np.array([func(doc) for doc in docs])
    return wrapper

