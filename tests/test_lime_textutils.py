# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
from hypothesis import given, assume
from hypothesis.strategies import characters, text

from eli5.lime.textutils import SplitResult, TokenizedText


def test_split_result():
    s = SplitResult.fromtext("")
    assert list(s.tokens) == []

    s = SplitResult.fromtext("hello")
    assert list(s.tokens) == ["hello"]

    s = SplitResult.fromtext("hello  world!")
    assert list(s.tokens) == ["hello", "world"]
    assert list(s.separators) == ["", "  ", "!"]


@given(text())
def test_split_result_combine(text):
    assume("\x00" not in text)  # fixme

    s = SplitResult.fromtext(text)
    assert s.text == text

    s_copy = s.copy()
    assert (s_copy.parts == s.parts).all()
    assert s_copy.parts is not s.parts


def test_split_result_masked():
    s = SplitResult.fromtext("Hello, world!")
    assert s.masked(np.array([False, False], dtype=bool)).text == s.text
    assert s.masked(np.array([True, False], dtype=bool)).text == ", world!"
    assert s.masked(np.array([False, True], dtype=bool)).text == "Hello, !"
    assert s.masked(np.array([True, True], dtype=bool)).text == ", !"


def test_token_spans():
    s = SplitResult.fromtext("Hello, world!")
    assert s.token_spans == [(0, 5), (7, 12)]
