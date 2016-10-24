import re

from six.moves import xrange
from sklearn.feature_extraction.text import VectorizerMixin

from eli5.sklearn.unhashing import InvertableHashingVectorizer


def get_weighted_spans(doc, vec, feature_weights):
    """ If possible, return a dict with preprocessed document and a list
    of spans with weights, corresponding to features in the document.
    """
    if isinstance(vec, InvertableHashingVectorizer):
        vec = vec.vec
    if not isinstance(vec, VectorizerMixin):
        return

    def _get_features(feature):
        if isinstance(feature, list):
            return [f['name'] for f in feature]
        else:
            return [feature]

    feature_weights_dict = {
        f: weight for group in ['pos', 'neg']
        for feature, weight in feature_weights[group]
        for f in _get_features(feature)}

    span_analyzer, preprocessed_doc = _build_span_analyzer(doc, vec)
    if span_analyzer is None:
        return
    weighted_spans = []
    found_features = set()
    for spans, feature in span_analyzer(preprocessed_doc):
        weight = feature_weights_dict.get(feature)
        if weight is not None:
            found_features.add(feature)
            weighted_spans.append((feature, spans, weight))
    not_found_items = sorted(
        ((feature, weight) for feature, weight in feature_weights_dict.items()
            if feature not in found_features),
        key=lambda x: x[1], reverse=True)
    not_found = {
        'pos': [(f, w) for f, w in not_found_items if w > 0],
        'neg': [(f, w) for f, w in not_found_items if w < 0],
    }
    return {
        'analyzer': vec.analyzer,
        'document': preprocessed_doc,
        'weighted_spans': weighted_spans,
        'not_found': not_found,
    }


def _build_span_analyzer(document, vec):
    """ Return an analyzer and the preprocessed doc.
    Analyzer will yield pairs of spans and feature, where spans are pairs
    of indices into the preprocessed doc. The idea here is to do minimal
    preprocessing so that we can still recover the same features as sklearn
    vectorizers, but with spans, that will allow us to highlight
    features in preprocessed documents.
    Analyzers are adapter from VectorizerMixin from sklearn.
    """
    preprocessed_doc = vec.build_preprocessor()(vec.decode(document))
    analyzer = None
    if vec.analyzer == 'word' and vec.tokenizer is None:
        stop_words = vec.get_stop_words()
        tokenize = _build_tokenizer(vec)
        analyzer = lambda doc: _word_ngrams(vec, tokenize(doc), stop_words)
    elif vec.analyzer == 'char':
        preprocessed_doc = vec._white_spaces.sub(' ', preprocessed_doc)
        analyzer = lambda doc: _char_ngrams(vec, doc)
    elif vec.analyzer == 'char_wb':
        preprocessed_doc = vec._white_spaces.sub(' ', preprocessed_doc)
        analyzer = lambda doc: _char_wb_ngrams(vec, doc)
    return analyzer, preprocessed_doc


# Adapted from VectorizerMixin.build_tokenizer

def _build_tokenizer(vec):
    token_pattern = re.compile(vec.token_pattern)
    tokenizer = lambda doc: [
        (m.span(), m.group()) for m in re.finditer(token_pattern, doc)]
    return tokenizer


# Adapted from VectorizerMixin._word_ngrams

def _word_ngrams(vec, tokens, stop_words=None):
    if stop_words is not None:
        tokens = [(s, w) for s, w in tokens if w not in stop_words]
    min_n, max_n = vec.ngram_range
    if max_n == 1:
        tokens = [([s], w) for s, w in tokens]
    else:
        original_tokens = tokens
        tokens = []
        n_original_tokens = len(original_tokens)
        for n in xrange(min_n,
                        min(max_n + 1, n_original_tokens + 1)):
            for i in xrange(n_original_tokens - n + 1):
                ngram_tokens = original_tokens[i: i + n]
                tokens.append((
                    [s for s, _ in ngram_tokens],
                    ' '.join(t for _, t in ngram_tokens)))
    return tokens


# Adapted from VectorizerMixin._char_wb_ngrams

def _char_ngrams(vec, text_document):
    text_len = len(text_document)
    ngrams = []
    min_n, max_n = vec.ngram_range
    for n in xrange(min_n, min(max_n + 1, text_len + 1)):
        for i in xrange(text_len - n + 1):
            ngrams.append(([(i, i + n)], text_document[i: i + n]))
    return ngrams


# Adapted from VectorizerMixin._char_wb_ngrams

def _char_wb_ngrams(vec, text_document):
    min_n, max_n = vec.ngram_range
    ngrams = []
    for m in re.finditer(r'\S+', text_document):
        w_start, w_end = m.start(), m.end()
        w = m.group(0)
        w = ' ' + w + ' '
        w_len = len(w)
        for n in xrange(min_n, max_n + 1):
            offset = 0
            ngrams.append((
                [(w_start + offset - 1, w_start + offset + n - 1)],
                w[offset:offset + n]))
            while offset + n < w_len:
                offset += 1
                ngrams.append((
                    [(w_start + offset - 1, w_start + offset + n - 1)],
                    w[offset:offset + n]))
            if offset == 0:   # count a short word (w_len < n) only once
                break
    return ngrams
