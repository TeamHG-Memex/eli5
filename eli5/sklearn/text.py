import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from six.moves import xrange
from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.pipeline import FeatureUnion

from eli5.base import (
    DocWeightedSpans, WeightedSpans, FeatureWeights, FeatureWeight,
    TargetExplanation)
from eli5.sklearn.unhashing import InvertableHashingVectorizer
from eli5.formatters import FormattedFeatureName


def get_weighted_spans(doc, vec, feature_weights):
    # type: (Any, Any, FeatureWeights) -> Optional[WeightedSpans]
    """ If possible, return a dict with preprocessed document and a list
    of spans with weights, corresponding to features in the document.
    """
    if isinstance(vec, FeatureUnion):
        return _get_weighted_spans_from_union(doc, vec, feature_weights)
    else:
        result = _get_doc_weighted_spans(doc, vec, feature_weights)
        if result is not None:
            found_features, doc_weighted_spans = result
            return WeightedSpans(
                [doc_weighted_spans],
                other=_get_other(feature_weights, [('', found_features)]),
            )


def add_weighted_spans(doc, vec, vectorized, target_expl):
    # type: (Any, Any, bool, TargetExplanation) -> None
    """
    Compute and set ``target_expl.weighted_spans`` attribute, when possible.
    """
    if vec is None or vectorized:
        return

    weighted_spans = get_weighted_spans(doc, vec, target_expl.feature_weights)
    if weighted_spans:
        target_expl.weighted_spans = weighted_spans


FoundFeatures = Dict[Tuple[str, int], float]


def _get_doc_weighted_spans(doc, vec, feature_weights, feature_fn=None):
    # type: (Any, Any, FeatureWeights, Callable[[str], str]) -> Optional[Tuple[FoundFeatures, DocWeightedSpans]]
    if isinstance(vec, InvertableHashingVectorizer):
        vec = vec.vec
    if not isinstance(vec, VectorizerMixin):
        return None

    span_analyzer, preprocessed_doc = _build_span_analyzer(doc, vec)
    if span_analyzer is None:
        return None

    # (group, idx) is a feature key here
    feature_weights_dict = {
        f: (fw.weight, (group, idx)) for group in ['pos', 'neg']
        for idx, fw in enumerate(getattr(feature_weights, group))
        for f in _get_features(fw.feature, feature_fn)}

    spans = []
    found_features = {}
    for f_spans, feature in span_analyzer(preprocessed_doc):
        try:
            weight, key = feature_weights_dict[feature]
        except KeyError:
            pass
        else:
            spans.append((feature, f_spans, weight))
            found_features[key] = weight

    return found_features, DocWeightedSpans(
        document=preprocessed_doc,
        spans=spans,
        preserve_density=vec.analyzer.startswith('char'),
    )


def _get_features(feature, feature_fn=None):
    if isinstance(feature, list):
        features = [f['name'] for f in feature]
    else:
        features = [feature]
    if feature_fn:
        features = list(filter(None, map(feature_fn, features)))
    return features


def _get_weighted_spans_from_union(doc, vec_union, feature_weights):
    # type: (Any, FeatureUnion, FeatureWeights) -> Optional[WeightedSpans]
    docs_weighted_spans = []
    named_found_features = []
    for vec_name, vec in vec_union.transformer_list:
        vec_prefix = '{}__'.format(vec_name)

        def feature_fn(x):
            if (not isinstance(x, FormattedFeatureName)
                    and x.startswith(vec_prefix)):
                return x[len(vec_prefix):]

        result = _get_doc_weighted_spans(doc, vec, feature_weights, feature_fn)
        if result:
            found_features, doc_weighted_spans = result
            doc_weighted_spans.vec_name = vec_name
            named_found_features.append((vec_name, found_features))
            docs_weighted_spans.append(doc_weighted_spans)

    if docs_weighted_spans:
        return WeightedSpans(
            docs_weighted_spans,
            other=_get_other(feature_weights, named_found_features),
        )


def _get_other(feature_weights, named_found_features):
    # type: (FeatureWeights, List[Tuple[str, FoundFeatures]]) -> FeatureWeights
    # search for items that were not accounted at all.
    other_items = []
    accounted_keys = set()  # type: Set[Tuple[str, int]]
    all_found_features = {}
    for _, found_features in named_found_features:
        all_found_features.update(found_features)

    for group in ['pos', 'neg']:
        for idx, fw in enumerate(getattr(feature_weights, group)):
            key = (group, idx)
            if key not in all_found_features and key not in accounted_keys:
                other_items.append(fw)
                accounted_keys.add(key)

    for vec_name, found_features in named_found_features:
        if found_features:
            other_items.append(FeatureWeight(
                feature=FormattedFeatureName(
                    '{}Highlighted in text (sum)'.format(
                        '{}: '.format(vec_name) if vec_name else '')),
                weight=sum(found_features.values())))

    other_items.sort(key=lambda x: abs(x.weight), reverse=True)
    return FeatureWeights(
        pos=[fw for fw in other_items if fw.weight >= 0],
        neg=[fw for fw in other_items if fw.weight < 0],
        pos_remaining=feature_weights.pos_remaining,
        neg_remaining=feature_weights.neg_remaining,
    )


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
