from sklearn.feature_extraction.text import CountVectorizer

from eli5.base import WeightedSpans, FeatureWeights
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
            pos=[('see', 2), ('lemon', 4), ('bias', 8)],
            neg=[('tree', -6)],
            neg_remaining=10
        ))
    assert w_spans == WeightedSpans(
        analyzer='word',
        document='i see: a leaning lemon tree',
        weighted_spans=[
            ('see', [(2, 5)], 2),
            ('lemon', [(17, 22)], 4),
            ('tree', [(23, 27)], -6)],
        other=FeatureWeights(
            pos=[('bias', 8), (hl_in_text, 0)],
            neg=[],
            neg_remaining=10,
        ))


def test_weighted_spans_word_bigrams():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    vec.fit([doc])
    w_spans = get_weighted_spans(
        doc, vec,
        FeatureWeights(
            pos=[('see', 2), ('leaning lemon', 5), ('lemon tree', 8)],
            neg=[('tree', -6)]))
    assert w_spans == WeightedSpans(
        analyzer='word',
        document='i see: a leaning lemon tree',
        weighted_spans=[
            ('see', [(2, 5)], 2),
            ('tree', [(23, 27)], -6),
            ('leaning lemon', [(9, 16), (17, 22)], 5),
            ('lemon tree', [(17, 22), (23, 27)], 8)],
        other=FeatureWeights(
            pos=[(hl_in_text, 9)],
            neg=[],
        ))


def test_weighted_spans_word_stopwords():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='word', stop_words='english')
    vec.fit([doc])
    w_spans = get_weighted_spans(
        doc, vec,
        FeatureWeights(
            pos=[('see', 2), ('lemon', 5), ('bias', 8)],
            neg=[('tree', -6)]))
    assert w_spans == WeightedSpans(
        analyzer='word',
        document='i see: a leaning lemon tree',
        weighted_spans=[
            ('lemon', [(17, 22)], 5),
            ('tree', [(23, 27)], -6)],
        other=FeatureWeights(
            pos=[('bias', 8), ('see', 2)],
            neg=[(hl_in_text, -1)],
        ))


def test_weighted_spans_char():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='char', ngram_range=(3, 4))
    vec.fit([doc])
    w_spans = get_weighted_spans(
        doc, vec,
        FeatureWeights(
            pos=[('see', 2), ('a le', 5), ('on ', 8)],
            neg=[('lem', -6)]))
    assert w_spans == WeightedSpans(
        analyzer='char',
        document='i see: a leaning lemon tree',
        weighted_spans=[
            ('see', [(2, 5)], 2),
            ('lem', [(17, 20)], -6),
            ('on ', [(20, 23)], 8),
            ('a le', [(7, 11)], 5)],
        other=FeatureWeights(
            pos=[(hl_in_text, 9)],
            neg=[],
        ))


def test_no_weighted_spans():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='char', ngram_range=(3, 4))
    vec.fit([doc])
    w_spans = get_weighted_spans(doc, vec, FeatureWeights(pos=[], neg=[]))
    assert w_spans == WeightedSpans(
        analyzer='char',
        document='i see: a leaning lemon tree',
        weighted_spans=[],
        other=FeatureWeights(pos=[], neg=[]))


def test_weighted_spans_char_wb():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='char_wb', ngram_range=(3, 4))
    vec.fit([doc])
    w_spans = get_weighted_spans(
        doc, vec,
        FeatureWeights(
            pos=[('see', 2), ('a le', 5), ('on ', 8)],
            neg=[('lem', -6), (' lem', -4)]))
    assert w_spans == WeightedSpans(
        analyzer='char_wb',
        document='i see: a leaning lemon tree',
        weighted_spans=[
            ('see', [(2, 5)], 2),
            ('lem', [(17, 20)], -6),
            ('on ', [(20, 23)], 8),
            (' lem', [(16, 20)], -4)],
        other=FeatureWeights(
            pos=[('a le', 5), (hl_in_text, 0)],
            neg=[],
        ))


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
                ([{'name': 'foo', 'sign': 1}, {'name': 'see', 'sign': -1}], 2),
                ([{'name': 'zoo', 'sign': 1}, {'name': 'bar', 'sign': 1}], 3),
            ],
            neg=[
                ([{'name': 'ree', 'sign': 1}, {'name': 'tre', 'sign': 1}], -4),
            ],
        ))
    assert w_spans == WeightedSpans(
        analyzer='char',
        document='i see: a leaning lemon tree',
        weighted_spans=[
            ('see', [(2, 5)], 2),
            ('tre', [(23, 26)], -4),
            ('ree', [(24, 27)], -4),
            ],
        other=FeatureWeights(
            pos=[
                ([{'name': 'zoo', 'sign': 1}, {'name': 'bar', 'sign': 1}], 3),
            ],
            neg=[(hl_in_text, -2)],
        ))
