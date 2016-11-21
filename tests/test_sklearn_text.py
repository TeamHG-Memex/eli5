from sklearn.feature_extraction.text import CountVectorizer

from eli5.base import WeightedSpans, FeatureWeights, FeatureWeight as FW
from eli5.formatters import FormattedFeatureName
from eli5.sklearn.text import get_weighted_spans


hl_in_text = FormattedFeatureName('Highlighted in text (sum)')


def test_weighted_spans_word():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='word')
    vec.fit([doc])
    w_spans = get_weighted_spans(
        doc, vec,
        FeatureWeights(
            pos=[FW('see', 2), FW('lemon', 4), FW('bias', 8)],
            neg=[FW('tree', -6)],
            neg_remaining=10
        ))
    assert w_spans == [WeightedSpans(
        analyzer='word',
        document='i see: a leaning lemon tree',
        weighted_spans=[
            ('see', [(2, 5)], 2),
            ('lemon', [(17, 22)], 4),
            ('tree', [(23, 27)], -6)],
        other=FeatureWeights(
            pos=[FW('bias', 8), FW(hl_in_text, 0)],
            neg=[],
            neg_remaining=10,
        ))]


def test_weighted_spans_word_bigrams():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    vec.fit([doc])
    w_spans = get_weighted_spans(
        doc, vec,
        FeatureWeights(
            pos=[FW('see', 2), FW('leaning lemon', 5), FW('lemon tree', 8)],
            neg=[FW('tree', -6)]))
    assert w_spans == [WeightedSpans(
        analyzer='word',
        document='i see: a leaning lemon tree',
        weighted_spans=[
            ('see', [(2, 5)], 2),
            ('tree', [(23, 27)], -6),
            ('leaning lemon', [(9, 16), (17, 22)], 5),
            ('lemon tree', [(17, 22), (23, 27)], 8)],
        other=FeatureWeights(
            pos=[FW(hl_in_text, 9)],
            neg=[],
        ))]


def test_weighted_spans_word_stopwords():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='word', stop_words='english')
    vec.fit([doc])
    w_spans = get_weighted_spans(
        doc, vec,
        FeatureWeights(
            pos=[FW('see', 2), FW('lemon', 5), FW('bias', 8)],
            neg=[FW('tree', -6)]))
    assert w_spans == [WeightedSpans(
        analyzer='word',
        document='i see: a leaning lemon tree',
        weighted_spans=[
            ('lemon', [(17, 22)], 5),
            ('tree', [(23, 27)], -6)],
        other=FeatureWeights(
            pos=[FW('bias', 8), FW('see', 2)],
            neg=[FW(hl_in_text, -1)],
        ))]


def test_weighted_spans_char():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='char', ngram_range=(3, 4))
    vec.fit([doc])
    w_spans = get_weighted_spans(
        doc, vec,
        FeatureWeights(
            pos=[FW('see', 2), FW('a le', 5), FW('on ', 8)],
            neg=[FW('lem', -6)]))
    assert w_spans == [WeightedSpans(
        analyzer='char',
        document='i see: a leaning lemon tree',
        weighted_spans=[
            ('see', [(2, 5)], 2),
            ('lem', [(17, 20)], -6),
            ('on ', [(20, 23)], 8),
            ('a le', [(7, 11)], 5)],
        other=FeatureWeights(
            pos=[FW(hl_in_text, 9)],
            neg=[],
        ))]


def test_no_weighted_spans():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='char', ngram_range=(3, 4))
    vec.fit([doc])
    w_spans = get_weighted_spans(doc, vec, FeatureWeights(pos=[], neg=[]))
    assert w_spans == [WeightedSpans(
        analyzer='char',
        document='i see: a leaning lemon tree',
        weighted_spans=[],
        other=FeatureWeights(pos=[], neg=[]))]


def test_weighted_spans_char_wb():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='char_wb', ngram_range=(3, 4))
    vec.fit([doc])
    w_spans = get_weighted_spans(
        doc, vec,
        FeatureWeights(
            pos=[FW('see', 2), FW('a le', 5), FW('on ', 8)],
            neg=[FW('lem', -6), FW(' lem', -4)]))
    assert w_spans == [WeightedSpans(
        analyzer='char_wb',
        document='i see: a leaning lemon tree',
        weighted_spans=[
            ('see', [(2, 5)], 2),
            ('lem', [(17, 20)], -6),
            ('on ', [(20, 23)], 8),
            (' lem', [(16, 20)], -4)],
        other=FeatureWeights(
            pos=[FW('a le', 5), FW(hl_in_text, 0)],
            neg=[],
        ))]


def test_unhashed_features_other():
    """ Check that when there are several candidates, they do not appear in "other"
    if at least one is found. If none are found, they should appear in "other"
    together.
    """
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='char', ngram_range=(3, 3))
    vec.fit([doc])
    w_spans = get_weighted_spans(
        doc, vec,
        FeatureWeights(
            pos=[
                FW([{'name': 'foo', 'sign': 1}, {'name': 'see', 'sign': -1}], 2),
                FW([{'name': 'zoo', 'sign': 1}, {'name': 'bar', 'sign': 1}], 3),
            ],
            neg=[
                FW([{'name': 'ree', 'sign': 1}, {'name': 'tre', 'sign': 1}], -4),
            ],
        ))
    assert w_spans == [WeightedSpans(
        analyzer='char',
        document='i see: a leaning lemon tree',
        weighted_spans=[
            ('see', [(2, 5)], 2),
            ('tre', [(23, 26)], -4),
            ('ree', [(24, 27)], -4),
            ],
        other=FeatureWeights(
            pos=[
                FW([{'name': 'zoo', 'sign': 1}, {'name': 'bar', 'sign': 1}], 3),
            ],
            neg=[FW(hl_in_text, -2)],
        ))]
