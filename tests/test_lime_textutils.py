# -*- coding: utf-8 -*-
from __future__ import absolute_import

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

