# -*- coding: utf-8 -*-

import pytest
import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

from eli5.sklearn.score_decrease import ScoreDecreaseFeatureImportances


def _boston_with_leak(X, y, feat_names, noise_scale=10.0, noise_ratio=0.25):
    rng = np.random.RandomState(42)
    noise = noise_scale * (rng.random_sample(y.shape).reshape(-1, 1) - 0.5)
    (X_train, X_test,
     y_train, y_test,
     noise_train, noise_test) = train_test_split(X, y, noise, random_state=42,
                                                 test_size=noise_ratio)

    X_train = np.hstack([X_train, noise_train + y_train.reshape(-1, 1)])
    X_test = np.hstack([X_test, noise_test])

    feat_names = list(feat_names) + ["DATALEAK"]
    return X_train, X_test, y_train, y_test, feat_names


def _assert_importances_good(sd, feat_names):
    importances = dict(zip(feat_names, sd.feature_importances_))
    print(importances)
    assert importances['LSTAT'] > importances['NOX']
    assert importances['B'] > importances['CHAS']
    return importances


def _assert_importances_not_overfit(sd, feat_names):
    importances = _assert_importances_good(sd, feat_names)
    assert importances['LSTAT'] > importances['DATALEAK']


def _assert_importances_overfit(sd, feat_names):
    importances = _assert_importances_good(sd, feat_names)
    assert importances['LSTAT'] < importances['DATALEAK']


def test_score_decrease_prefit(boston_train):
    X_train, X_test, y_train, y_test, feat_names = _boston_with_leak(
        *boston_train)

    # prefit estimator, fit on train part
    reg = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)
    print(reg.score(X_test, y_test), reg.score(X_train, y_train))
    print(X_train[:,-1], y_train)
    sd_prefit = ScoreDecreaseFeatureImportances(reg, random_state=42, n_iter=10).fit(X_train, y_train)
    assert not hasattr(sd_prefit, "estimator_")
    assert sd_prefit.wrapped_estimator_ is reg
    _assert_importances_overfit(sd_prefit, feat_names)

    # prefit estimator, fit on test part
    sd_prefit2 = ScoreDecreaseFeatureImportances(reg, random_state=42, n_iter=10).fit(X_test, y_test)
    _assert_importances_not_overfit(sd_prefit2, feat_names)


def test_score_decrease_cv(boston_train):
    # noise feature can be important if no cv is used, but not if cv is used
    X_train, X_test, y_train, y_test, feat_names = _boston_with_leak(
        *boston_train, noise_ratio=0.99)

    reg = ScoreDecreaseFeatureImportances(
        SVR(C=100),
        random_state=42,
        cv=None
    ).fit(X_test, y_test)

    assert reg.score(X_test, y_test) > 0
    assert reg.estimator_.score(X_test, y_test) > 0
    print(reg.score(X_test, y_test))
    imp_nocv = _assert_importances_good(reg, feat_names)

    # CV feature importances
    reg = ScoreDecreaseFeatureImportances(
        SVR(C=100),
        random_state=42,
        cv=10,
    ).fit(X_test, y_test)
    imp_cv = _assert_importances_good(reg, feat_names)
    assert reg.score(X_test, y_test) > 0

    assert imp_cv['DATALEAK'] < imp_nocv['DATALEAK']


def test_score_decrease_params():
    with pytest.raises(ValueError):
        reg = ScoreDecreaseFeatureImportances(SVR(), cv="hello")


def test_score_decrease_classifier(iris_train):
    X, y, feature_names, target_names = iris_train
    clf = LogisticRegression().fit(X, y)
    assert is_classifier(clf)
    sd = ScoreDecreaseFeatureImportances(clf, random_state=42).fit(X, y)
    assert is_classifier(sd)
    assert (sd.classes_ == [0, 1, 2]).all()
    assert np.allclose(clf.predict(X), sd.predict(X))
    assert np.allclose(clf.predict_proba(X), sd.predict_proba(X))
    assert np.allclose(clf.predict_log_proba(X), sd.predict_log_proba(X))
    assert np.allclose(clf.decision_function(X), sd.decision_function(X))


def test_score_decrease_type():
    sd = ScoreDecreaseFeatureImportances(LogisticRegression(), cv=3)
    assert is_classifier(sd)

    sd = ScoreDecreaseFeatureImportances(RandomForestRegressor(), cv=3)
    assert is_regressor(sd)


def test_feature_selection(boston_train):
    X, y, feature_names = boston_train

    sel = SelectFromModel(
        ScoreDecreaseFeatureImportances(
            RandomForestRegressor(n_estimators=20),
            cv=3, random_state=42, refit=False
        ),
        threshold=0.1,
    )
    pipe = make_pipeline(sel, SVR(C=10))
    score1 = cross_val_score(pipe, X, y).mean()
    score2 = cross_val_score(SVR(C=10), X, y).mean()
    print(score1, score2)
    assert score1 > score2

    sel.fit(X, y)
    selected = {feature_names[idx] for idx in sel.get_support(indices=True)}
    assert selected == {'LSTAT', 'RM'}
