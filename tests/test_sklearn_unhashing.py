from itertools import repeat
from collections import Counter

import pytest
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from eli5.sklearn.unhashing import InvertableHashingVectorizer
from eli5.sklearn.utils import sklearn_version

@pytest.mark.parametrize(
    ['always_signed', 'binary', 'alternate_sign'], [
        [True, False, True],
        [False, False, True],
        [False, True, True],
        [True, True, True],
        [False, True, False],
        [False, False, False],
        [True, False, False],
    ],
)
def test_invertable_hashing_vectorizer(always_signed, binary, alternate_sign):
    n_features = 8
    n_words = 4 * n_features
    kwargs = dict(n_features=n_features, binary=binary)
    if sklearn_version() < '0.19':
        kwargs['non_negative'] = not alternate_sign
    else:
        kwargs['alternate_sign'] = alternate_sign
    vec = HashingVectorizer(**kwargs)
    words = ['word_{}'.format(i) for i in range(n_words)]
    corpus = [w for i, word in enumerate(words, 1) for w in repeat(word, i)]
    split = len(corpus) // 2
    doc1, doc2 = ' '.join(corpus[:split]), ' '.join(corpus[split:])

    ivec = InvertableHashingVectorizer(vec)
    ivec.fit([doc1, doc2])
    check_feature_names(vec, ivec, always_signed, corpus, alternate_sign)

    ivec = InvertableHashingVectorizer(vec)
    ivec.partial_fit([doc1])
    ivec.partial_fit([doc2])
    check_feature_names(vec, ivec, always_signed, corpus, alternate_sign)

    ivec = InvertableHashingVectorizer(vec)
    for w in corpus:
        ivec.partial_fit([w])
    check_feature_names(vec, ivec, always_signed, corpus, alternate_sign)


def check_feature_names(vec, ivec, always_signed, corpus, alternate_sign):
    feature_names = ivec.get_feature_names(always_signed=always_signed)
    seen_words = set()
    counts = Counter(corpus)
    for idx, collisions in enumerate(feature_names):
        words_in_collision = []
        for ic, collision in enumerate(collisions):
            sign = collision['sign']
            c = collision['name']
            if ic == 0 and not always_signed:
                # Most frequent term is always not inverted.
                assert sign == 1, collisions
            seen_words.add(c)
            words_in_collision.append(c)
            if not always_signed and ivec.column_signs_[idx] < 0:
                sign *= -1
            # Term hashes to correct value with correct sign.
            expected = np.zeros(vec.n_features)
            expected[idx] = sign
            transormed = vec.transform([c]).toarray()
            assert np.allclose(transormed, expected), (transormed, expected)
        for prev_w, w in zip(words_in_collision, words_in_collision[1:]):
            # Terms are ordered by frequency.
            assert counts[prev_w] > counts[w]

    if not alternate_sign:
        assert np.array_equal(ivec.column_signs_, np.ones(len(feature_names)))
    assert seen_words == set(corpus)
