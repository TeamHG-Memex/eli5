# -*- coding: utf-8 -*-
"""
Utilities for text generation.
"""
from __future__ import absolute_import
import re
from typing import List, Tuple, Any, Union

import numpy as np
from sklearn.utils import check_random_state

from eli5.utils import indices_to_bool_mask, vstack


# the same as scikit-learn token pattern, but allows single-char tokens
DEFAULT_TOKEN_PATTERN = r'(?u)\b\w+\b'


def generate_samples(text, n_samples=500, bow=True, random_state=None):
    # type: (TokenizedText, int, bool, Any) -> Tuple[List[str], np.ndarray, np.ndarray]
    """
    Return ``n_samples`` changed versions of text (with some words removed),
    along with distances between the original text and a generated
    examples. If ``bow=False``, all tokens are considered unique
    (i.e. token position matters).
    """
    if bow:
        num_tokens = len(text.vocab)
        res = text.replace_random_tokens_bow(n_samples,
                                             random_state=random_state)
    else:
        num_tokens = len(text.tokens)
        res = text.replace_random_tokens(n_samples, random_state=random_state)

    texts, num_removed_vec, masks = zip(*res)
    similarity = cosine_similarity_vec(num_tokens, num_removed_vec)
    return texts, similarity, vstack(masks)


def cosine_similarity_vec(num_tokens, num_removed_vec):
    """
    Return cosine similarity between a binary vector with all ones
    of length ``num_tokens`` and vectors of the same length with
    ``num_removed_vec`` elements set to zero.
    """
    remaining = -np.array(num_removed_vec) + num_tokens
    return remaining / (np.sqrt(num_tokens + 1e-6) * np.sqrt(remaining + 1e-6))


class TokenizedText(object):
    def __init__(self, text, token_pattern=DEFAULT_TOKEN_PATTERN):
        self.text = text
        self.split = SplitResult.fromtext(text, token_pattern)
        self._vocab = None  # type: List[str]

    def replace_random_tokens(self, n_samples, replacement='',
                              random_state=None):
        """ 
        Return a list of ``(text, replaced_count, mask)``
        tuples with n_samples versions of text with some words replaced.
        By default words are replaced with '', i.e. removed.
        """
        n_tokens = len(self.tokens)
        indices = np.arange(n_tokens)
        if not n_tokens:
            nomask = np.array([], dtype=int)
            return [('', 0, nomask)] * n_samples
        rng = check_random_state(random_state)
        sizes = rng.randint(low=1, high=n_tokens + 1, size=n_samples)
        res = []
        for size in sizes:
            to_remove = rng.choice(indices, size, replace=False)
            mask = indices_to_bool_mask(to_remove, n_tokens)
            s = self.split.masked(to_remove, replacement)
            res.append((s.text, size, mask))
        return res
    
    def replace_random_tokens_bow(self, n_samples, replacement='',
                                  random_state=None):
        """ 
        Return a list of ``(text, replaced_words_count, mask)`` tuples with
        n_samples versions of text with some words replaced.
        If a word is replaced, all duplicate words are also replaced
        from the text. By default words are replaced with '', i.e. removed.
        """
        if not self.vocab:
            nomask = np.array([], dtype=int)
            return [('', 0, nomask)] * n_samples

        rng = check_random_state(random_state)
        sizes = rng.randint(low=1, high=len(self.vocab) + 1, size=n_samples)
        res = []
        for size in sizes:
            tokens_to_remove = set(rng.choice(self.vocab, size, replace=False))
            to_remove = [idx for idx, token in enumerate(self.tokens)
                         if token in tokens_to_remove]
            mask = indices_to_bool_mask(to_remove, len(self.tokens))
            s = self.split.masked(to_remove, replacement)
            res.append((s.text, size, mask))
        return res

    @property
    def vocab(self):
        # type: () -> List[str]
        if self._vocab is None:
            self._vocab = list(set(self.tokens))
        return self._vocab

    @property
    def tokens(self):
        return self.split.tokens


class SplitResult(object):
    def __init__(self, parts):
        self.parts = np.array(parts, ndmin=1)

    @classmethod
    def fromtext(cls, text, token_pattern=DEFAULT_TOKEN_PATTERN):
        token_pattern = u"(%s)" % token_pattern
        parts = re.split(token_pattern, text)
        return cls(parts)

    @property
    def separators(self):
        return self.parts[::2]

    @property
    def tokens(self):
        return self.parts[1::2]

    def copy(self):
        # type: () -> SplitResult
        return self.__class__(self.parts.copy())

    def masked(self, invmask, replacement=''):
        # type: (Union[np.ndarray, List[int]], str) -> SplitResult
        s = self.copy()
        s.tokens[invmask] = replacement
        return s

    @property
    def text(self):
        # type: () -> str
        return "".join(self.parts)
