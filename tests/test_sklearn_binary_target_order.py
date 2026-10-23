import json

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import eli5


@pytest.mark.parametrize('estimator', [
    LogisticRegression(solver='liblinear', random_state=0),
    LinearSVC(random_state=0),
    DecisionTreeClassifier(max_depth=1, random_state=0),
    RandomForestClassifier(n_estimators=3, max_depth=1, random_state=0,
                           n_jobs=1),
])
@pytest.mark.parametrize('value', [-2.0, 2.0])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('rename', [False, True])
def test_binary_prediction_target_order(estimator, value, reverse, rename):
    X = np.array([[-4.0], [-3.0], [-2.0], [-1.0],
                  [1.0], [2.0], [3.0], [4.0]])
    estimator.fit(X, ['ant'] * 4 + ['bee'] * 4)
    doc = np.array([value])
    names = {'ant': 'negative', 'bee': 'positive'} if rename else {}
    predicted = estimator.predict([doc])[0]
    expected = names.get(predicted, predicted)
    baseline = eli5.explain_prediction(estimator, doc, target_names=names)
    assert baseline.targets[0].target == expected

    targets = [names.get(label, label) for label in estimator.classes_]
    if reverse:
        targets.reverse()
    result = eli5.explain_prediction(estimator, doc, target_names=names,
                                    targets=targets)
    target = result.targets[0]
    assert target.target == expected
    assert target.feature_weights == baseline.targets[0].feature_weights
    if hasattr(estimator, 'predict_proba'):
        label_id = list(estimator.classes_).index(predicted)
        assert target.proba == pytest.approx(
            estimator.predict_proba([doc])[0, label_id])
    serialized = json.loads(json.dumps(eli5.format_as_dict(result)))
    assert serialized['targets'][0]['target'] == expected
