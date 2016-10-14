import re

import numpy as np
from six.moves import xrange
from sklearn.feature_extraction.text import VectorizerMixin


def highlighted_features(doc, vec, feature_weights):
    if not isinstance(vec, VectorizerMixin):
        return
    feature_weights_dict = {
        feature: weight for group in ['pos', 'neg']
        for feature, weight in feature_weights[group]}
    span_analyzer, preprocessed_doc = _build_span_analyzer(doc, vec)
    if span_analyzer is None:
        # TODO - fallback to work on tokens
        return
    weighted_spans = []
    for spans, feature in span_analyzer(preprocessed_doc):
        weight = feature_weights_dict.get(feature)
        if weight is not None:
            weighted_spans.append((spans, weight))
    return _highlight(preprocessed_doc, weighted_spans)


def _build_span_analyzer(document, vec):
    preprocessed_doc = vec.build_preprocessor()(vec.decode(document))
    analyzer = None
    if vec.analyzer == 'word' and vec.tokenizer is None:
        stop_words = vec.get_stop_words()
        tokenize = _build_tokenizer(vec)
        analyzer = lambda doc: _word_ngrams(vec, tokenize(doc), stop_words)
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


def _highlight(doc, weighted_spans):
    char_weights = np.zeros(len(doc))
    for spans, weight in weighted_spans:
        for start, end in spans:
            char_weights[start:end] += weight
    # TODO - can be much smarter, join spans at least
    # TODO - for longer documents, remove text without active features
    weight_range = max(abs(char_weights.min()), abs(char_weights.max()))
    return ''.join(
        c if np.isclose(weight, 0.) else
        '<span style="background-color: {color}" '
        'title="{weight:.3f}"'
        '>{c}</span>'.format(
            color=weight_color(weight, weight_range),
            weight=weight,
            c=c)
        for c, weight in zip(doc, char_weights))


def weight_color(weight, weight_range):
    alpha = (abs(weight) / weight_range) ** 1.5
    h, l = 255, 150
    if weight > 0:
        rgb = (l, h, l)
    else:
        rgb = (h, l, l)
    rbga = rgb + (alpha,)
    return 'rgba{}'.format(rbga)
