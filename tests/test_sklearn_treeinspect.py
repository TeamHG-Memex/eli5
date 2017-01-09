# -*- coding: utf-8 -*-
from __future__ import absolute_import

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from eli5.formatters.trees import tree2text
from eli5.sklearn.treeinspect import get_tree_info


def test_tree2dict():
    X = [[1, 1], [0, 2], [0, 3], [1, 3], [2, 3], [0, 4]]
    y = [0, 0, 0, 1, 1, 1]
    clf = DecisionTreeClassifier(random_state=42).fit(X, y)
    text = tree2text(get_tree_info(clf))
    print(text)
    expected = """
x1 <= 2.500  (33.3%)  ---> 0.000
x1 > 2.500  (66.7%)
    x0 <= 0.500  (33.3%)
        x1 <= 3.500  (16.7%)  ---> 0.000
        x1 > 3.500  (16.7%)  ---> 1.000
    x0 > 0.500  (33.3%)  ---> 1.000
""".strip()
    assert text == expected

    # check it with feature_names
    text = tree2text(get_tree_info(clf, feature_names=['x', 'y']))
    print(text)
    expected = """
y <= 2.500  (33.3%)  ---> 0.000
y > 2.500  (66.7%)
    x <= 0.500  (33.3%)
        y <= 3.500  (16.7%)  ---> 0.000
        y > 3.500  (16.7%)  ---> 1.000
    x > 0.500  (33.3%)  ---> 1.000
""".strip()
    assert text == expected

