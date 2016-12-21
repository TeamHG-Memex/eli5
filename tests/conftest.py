# -*- coding: utf-8 -*-
import pytest
import numpy as np
from sklearn.datasets import fetch_20newsgroups, load_boston
from sklearn.utils import shuffle

NEWSGROUPS_CATEGORIES = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
NEWSGROUPS_CATEGORIES_BINARY = [
    'alt.atheism',
    'comp.graphics',
]
SIZE = 100


def _get_newsgroups(binary=False, remove_chrome=False, test=False, size=SIZE):
    remove = ('headers', 'footers', 'quotes') if remove_chrome else []
    categories = (
        NEWSGROUPS_CATEGORIES_BINARY if binary else NEWSGROUPS_CATEGORIES)
    subset = 'test' if test else 'train'
    data = fetch_20newsgroups(subset=subset, categories=categories,
                              shuffle=True, random_state=42,
                              remove=remove)
    return data.data[:size], data.target[:size], data.target_names


@pytest.fixture(scope="session")
def newsgroups_train():
    return _get_newsgroups(remove_chrome=True)


@pytest.fixture(scope="session")
def newsgroups_train_binary():
    return _get_newsgroups(binary=True, remove_chrome=True)


@pytest.fixture(scope="session")
def newsgroups_train_binary_big():
    return _get_newsgroups(binary=True, remove_chrome=True, size=1000)


@pytest.fixture(scope="session")
def boston_train(size=SIZE):
    data = load_boston()
    X, y = shuffle(data.data, data.target, random_state=13)
    X = X.astype(np.float32)
    return X[:size], y[:size], data.feature_names
