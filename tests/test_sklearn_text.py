from sklearn.feature_extraction.text import CountVectorizer

from eli5.sklearn.text import get_weighted_spans


def test_weighted_spans_word():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='word')
    vec.fit([doc])
    w_spans = get_weighted_spans(
        doc, vec,
        {'pos': [('see', 0.2), ('lemon', 0.5), ('bias', 0.8)],
         'neg': [('tree', -0.6)]})
    assert w_spans == {
        'document': 'i see: a leaning lemon tree',
        'weighted_spans': [
            ([(2, 5)], 0.2),
            ([(17, 22)], 0.5),
            ([(23, 27)], -0.6)],
        'not_found': {'bias': 0.8}}


def test_weighted_spans_word_bigrams():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    vec.fit([doc])
    w_spans = get_weighted_spans(
        doc, vec,
        {'pos': [('see', 0.2), ('leaning lemon', 0.5), ('lemon tree', 0.8)],
         'neg': [('tree', -0.6)]})
    assert w_spans == {
        'document': 'i see: a leaning lemon tree',
        'weighted_spans': [
            ([(2, 5)], 0.2),
            ([(23, 27)], -0.6),
            ([(9, 16), (17, 22)], 0.5),
            ([(17, 22), (23, 27)], 0.8)],
        'not_found': {}}


def test_weighted_spans_word_stopwords():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='word', stop_words='english')
    vec.fit([doc])
    w_spans = get_weighted_spans(
        doc, vec,
        {'pos': [('see', 0.2), ('lemon', 0.5), ('bias', 0.8)],
         'neg': [('tree', -0.6)]})
    assert w_spans == {
        'document': 'i see: a leaning lemon tree',
        'weighted_spans': [
            ([(17, 22)], 0.5),
            ([(23, 27)], -0.6)],
        'not_found': {'bias': 0.8, 'see': 0.2}}


def test_weighted_spans_char():
    doc = 'I see: a leaning lemon tree'
    vec = CountVectorizer(analyzer='char', ngram_range=(3, 4))
    vec.fit([doc])
    w_spans = get_weighted_spans(
        doc, vec,
        {'pos': [('see', 0.2), ('a le', 0.5), ('on ', 0.8)],
         'neg': [('lem', -0.6)]})
    assert w_spans == {
        'document': 'i see: a leaning lemon tree',
        'weighted_spans': [
            ([(2, 5)], 0.2),
            ([(17, 20)], -0.6),
            ([(20, 23)], 0.8),
            ([(7, 11)], 0.5)],
        'not_found': {}}
