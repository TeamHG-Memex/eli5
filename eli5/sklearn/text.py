import re

from six.moves import xrange
from sklearn.feature_extraction.text import VectorizerMixin


def highlighted_features(doc, vec, feature_weights, window=5):
    if not isinstance(vec, VectorizerMixin):
        return
    feature_weights_dict = {
        feature: weight for group in ['pos', 'neg']
        for feature, weight in feature_weights[group]}
    span_analyzer = _build_span_analyzer(vec)
    if span_analyzer is None:
        # TODO - fallback to work on tokens
        return
    preprocessed_doc = vec.build_preprocessor()(vec.decode(doc))
    weighted_spans = []
    for span, feature in span_analyzer(preprocessed_doc):
        weight = feature_weights_dict.get(feature)
        if weight is not None:
            weighted_spans.append((span, weight))
    return weighted_spans


def _build_span_analyzer(vec):
    if vec.analyzer == 'word' and vec.tokenizer is None:
        stop_words = vec.get_stop_words()
        tokenize = _build_tokenizer(vec)
        return lambda preprocessed_doc: _word_ngrams(
            vec, tokenize(preprocessed_doc), stop_words)


def _build_tokenizer(vec):
    token_pattern = re.compile(vec.token_pattern)
    tokenizer = lambda doc: [
        (m.span(), m.group()) for m in re.finditer(token_pattern, doc)]
    return tokenizer


def _word_ngrams(vec, tokens, stop_words=None):
    if stop_words is not None:
        tokens = [w for w in tokens if w not in stop_words]
    min_n, max_n = vec.ngram_range
    if max_n != 1:
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
