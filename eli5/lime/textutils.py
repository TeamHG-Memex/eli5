# -*- coding: utf-8 -*-
"""
Utilities for text generation.
"""
from __future__ import absolute_import
import re
import random

import numpy as np


DEFAULT_TOKEN_PATTERN = r'(?u)\b\w+\b'


def generate_perturbations(text, n_samples=500, bow=True,
                           token_pattern=DEFAULT_TOKEN_PATTERN):
    """
    Return ``n_samples`` changed versions of text (with some words removed),
    along with distances between the original text and a generated
    examples. If ``bow=False``, all tokens are considered unique
    (i.e. token position matters).
    """
    t = TokenizedText(text, token_pattern=token_pattern)
    if bow:
        num_tokens = len(set(t.split.tokens))
        res = t.replace_random_tokens_bow(n_samples)
    else:
        num_tokens = len(t.split.tokens)
        res = t.replace_random_tokens(n_samples)

    texts, num_removed = zip(*res)
    similarity = cosine_similarity_vec(num_tokens, num_removed)
    return texts, similarity


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
        
    def replace_random_tokens(self, n_samples, replacement=''):
        """ 
        Return a list of ``(text, replaced_count)`` tuples with
        n_samples versions of text with some words replaced.
        By default words are replaced with '', i.e. removed.
        """
        indices = np.arange(len(self.split.tokens))
        sizes = np.random.randint(low=1, high=indices.shape[0], size=n_samples)
        res = []
        for size in sizes:
            s = self.split.copy()
            to_remove = np.random.choice(indices, size, replace=False)
            s.tokens[to_remove] = replacement
            res.append(("".join(s.parts), size))
        return res
    
    def replace_random_tokens_bow(self, n_samples, replacement=''):
        """ 
        Return a list of ``(text, replaced_words_count)`` tuples with
        n_samples versions of text with some words replaced.
        If a word is replaced, all duplicate words are also replaced
        from the text. By default words are replaced with '', i.e. removed.
        """
        vocab = set(self.split.tokens)
        sizes = np.random.randint(low=1, high=len(vocab), size=n_samples)
        res = []
        for size in sizes:
            to_remove = set(random.sample(vocab, size))
            parts = [
                p if p not in to_remove else replacement
                for p in self.split.parts
            ]
            res.append(("".join(parts), size))
        return res


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
        return SplitResult(self.parts.copy())

