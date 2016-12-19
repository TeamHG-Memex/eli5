# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from eli5.lime import get_local_pipeline_text
from eli5.sklearn import explain_prediction_sklearn
from eli5.formatters import format_as_text
from .utils import format_as_all


def test_lime_explain_probabilistic(newsgroups_train):
    docs, y, target_names = newsgroups_train
    vec = HashingVectorizer(non_negative=True)
    clf = MultinomialNB()

    X = vec.fit_transform(docs)
    clf.fit(X, y)

    pipe = make_pipeline(vec, clf)
    doc = docs[0]

    clf_local, vec_local, metrics = get_local_pipeline_text(doc,
                                                            pipe.predict_proba,
                                                            n_samples=10000,
                                                            expand_factor=10,
                                                            random_state=42)
    print(metrics)
    assert metrics['score'] > 0.7
    assert metrics['mean_KL_divergence'] < 0.1

    res = explain_prediction_sklearn(clf_local, doc, vec_local, top=20,
                                     target_names=target_names)
    expl = format_as_text(res)
    print(expl)
    assert 'file' in expl


def test_lime_flat_neighbourhood(newsgroups_train):
    docs, y, target_names = newsgroups_train
    doc = docs[0]

    def predict_proba(docs):
        """ This function predicts non-zero probabilities only for 3 labels """
        proba_graphics = [0, 1.0, 0, 0]
        proba_other = [0.9, 0, 0.1, 0]
        return np.array(
            [proba_graphics if 'file' in doc else proba_other for doc in docs]
        )


    clf_local, vec_local, metrics = get_local_pipeline_text(doc,
                                                            predict_proba,
                                                            n_samples=10000,
                                                            expand_factor=None,
                                                            random_state=42)
    print(metrics)
    print(clf_local.classes_, target_names)

    actual_target_names = np.array(target_names)[clf_local.classes_]
    res = explain_prediction_sklearn(clf_local, doc, vec_local, top=20,
                                     target_names=actual_target_names)
    for expl in format_as_all(res, clf_local):
        assert 'file' in expl
        assert "comp.graphics" in expl
