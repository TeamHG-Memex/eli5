# -*- coding: utf-8 -*-
import pytest
import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

import eli5
from eli5.sklearn import PermutationImportance
from .utils import format_as_all


def _boston_with_leak(X, y, feat_names, noise_scale=10.0, noise_ratio=0.25):
    rng = np.random.RandomState(42)
    noise = noise_scale * (rng.random_sample(y.shape).reshape(-1, 1) - 0.5)
    (X_train, X_test,
     y_train, y_test,
     noise_train, noise_test) = train_test_split(X, y, noise, random_state=42,
                                                 test_size=noise_ratio)

    # noise correlates with y in train, but not in test
    X_train = np.hstack([X_train, noise_train + y_train.reshape(-1, 1)])
    X_test = np.hstack([X_test, noise_test])

    feat_names = list(feat_names) + ["DATALEAK"]
    return X_train, X_test, y_train, y_test, feat_names


def _assert_importances_good(perm, feat_names):
    importances = dict(zip(feat_names, perm.feature_importances_))
    print(perm.scores_)
    print(importances)
    assert importances['LSTAT'] > importances['NOX']
    return importances


def _assert_importances_not_overfit(perm, feat_names):
    importances = _assert_importances_good(perm, feat_names)
    assert importances['LSTAT'] > importances['DATALEAK']


def _assert_importances_overfit(perm, feat_names):
    importances = _assert_importances_good(perm, feat_names)
    assert importances['LSTAT'] < importances['DATALEAK']


def test_prefit(boston_train):
    X_train, X_test, y_train, y_test, feat_names = _boston_with_leak(
        *boston_train)

    # prefit estimator, fit on train part
    reg = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)
    print(reg.score(X_test, y_test), reg.score(X_train, y_train))
    print(X_train[:, -1], y_train)
    perm_prefit = PermutationImportance(reg, random_state=42, n_iter=10).fit(X_train, y_train)
    assert not hasattr(perm_prefit, "estimator_")
    assert perm_prefit.wrapped_estimator_ is reg
    _assert_importances_overfit(perm_prefit, feat_names)

    # prefit estimator, fit on test part
    perm_prefit2 = PermutationImportance(reg, random_state=42, n_iter=10).fit(X_test, y_test)
    _assert_importances_not_overfit(perm_prefit2, feat_names)


def test_cv(boston_train):
    # noise feature can be important if no cv is used, but not if cv is used
    # X_train, y_train are almost empty; we're using test part of the dataset
    X_train, X_test, y_train, y_test, feat_names = _boston_with_leak(
        *boston_train, noise_ratio=0.99)

    reg = PermutationImportance(
        SVR(C=100),
        random_state=42,
        cv=None,
        n_iter=50,  # use the same number of experiments as with cv=10
    ).fit(X_test, y_test)

    assert reg.score(X_test, y_test) > 0
    assert reg.estimator_.score(X_test, y_test) > 0
    print(reg.score(X_test, y_test))
    imp_nocv = _assert_importances_good(reg, feat_names)

    # CV feature importances
    reg = PermutationImportance(
        SVR(C=100),
        random_state=42,
        cv=10,
    ).fit(X_test, y_test)
    imp_cv = _assert_importances_good(reg, feat_names)
    assert reg.score(X_test, y_test) > 0

    assert imp_cv['DATALEAK'] * 10 < imp_nocv['DATALEAK']


def test_invalid_params():
    with pytest.raises(ValueError):
        reg = PermutationImportance(SVR(), cv="hello")


def test_classifier(iris_train):
    X, y, feature_names, target_names = iris_train
    clf = LogisticRegression().fit(X, y)
    assert is_classifier(clf)
    perm = PermutationImportance(clf, random_state=42).fit(X, y)
    assert is_classifier(perm)
    assert (perm.classes_ == [0, 1, 2]).all()
    assert np.allclose(clf.predict(X), perm.predict(X))
    assert np.allclose(clf.predict_proba(X), perm.predict_proba(X))
    assert np.allclose(clf.predict_log_proba(X), perm.predict_log_proba(X))
    assert np.allclose(clf.decision_function(X), perm.decision_function(X))


def test_estimator_type():
    perm = PermutationImportance(LogisticRegression(), cv=3)
    assert is_classifier(perm)

    perm = PermutationImportance(RandomForestRegressor(), cv=3)
    assert is_regressor(perm)


def test_feature_selection(boston_train):
    X, y, feature_names = boston_train

    sel = SelectFromModel(
        PermutationImportance(
            RandomForestRegressor(n_estimators=20, random_state=42),
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


def test_explain_weights(iris_train):
    X, y, feature_names, target_names = iris_train
    kwargs = dict(n_iter=20, random_state=42)
    for perm in [
        PermutationImportance(SVC(C=10).fit(X, y), **kwargs),
        PermutationImportance(SVC(C=10), cv=None, **kwargs),
        PermutationImportance(SVC(C=10), cv=3, **kwargs),
    ]:
        perm.fit(X, y)
        print(perm.score(X, y))
        expl = eli5.explain_weights(perm, target_names=target_names,
                                    feature_names=feature_names)
        assert "generalization" in expl.description
        imp = expl.feature_importances.importances
        assert len(imp) == 4
        assert [n.feature.startswith("petal") for n in imp[:2]]
        assert [n.feature.startswith("sepal") for n in imp[2:]]

        res = format_as_all(expl, perm.wrapped_estimator_)
        for _expl in res:
            assert "petal width (cm)" in _expl

def test_pandas_xgboost_support(iris_train):
    xgboost = pytest.importorskip('xgboost')
    pd = pytest.importorskip('pandas')
    X, y, feature_names, target_names = iris_train
    X = pd.DataFrame(X)
    y = pd.Series(y)
    est = xgboost.XGBClassifier()
    est.fit(X, y)
    # we expect no exception to be raised here when using xgboost with pd.DataFrame
    perm = PermutationImportance(est).fit(X, y) 
