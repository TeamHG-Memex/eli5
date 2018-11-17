# -*- coding: utf-8 -*-
from __future__ import absolute_import
import abc
from functools import partial
from typing import List, Tuple, Any, Union, Dict, Optional
import six

import numpy as np  # type: ignore
from scipy.stats import itemfreq  # type: ignore
from sklearn.base import BaseEstimator, clone  # type: ignore
from sklearn.neighbors import KernelDensity  # type: ignore
from sklearn.metrics import pairwise_distances  # type: ignore
from sklearn.model_selection import GridSearchCV, KFold  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

from eli5.utils import vstack
from eli5.lime.utils import rbf
from .textutils import generate_samples, DEFAULT_TOKEN_PATTERN, TokenizedText


@six.add_metaclass(abc.ABCMeta)
class BaseSampler(BaseEstimator):
    """
    Base sampler class.
    Sampler is an object which generates examples similar to a given example.
    """
    @abc.abstractmethod
    def sample_near(self, doc, n_samples=1):
        """
        Return (examples, similarity) tuple with generated documents
        similar to a given document and a vector of similarity values.
        """
        raise NotImplementedError()

    def fit(self, X=None, y=None):
        return self


class MaskingTextSampler(BaseSampler):
    """
    Sampler for text data. It randomly removes or replaces tokens from text.

    Parameters
    ----------
    token_pattern : str, optional
        Regexp for token matching
    bow : bool, optional
        Sampler could either replace all instances of a given token
        (bow=True, bag of words sampling) or replace just a single token
        (bow=False).
    random_state : integer or numpy.random.RandomState, optional
        random state
    replacement : str
        Defalt value is '' - by default tokens are removed. If you want to
        preserve the total token count set ``replacement`` to a non-empty
        string, e.g. 'UNKN'.
    min_replace : int or float
        A minimum number of tokens to replace. Default is 1, meaning 1 token.
        If this value is float in range [0.0, 1.0], it is used as a ratio.
        More than min_replace tokens could be replaced if group_size > 1.
    max_replace : int or float
        A maximum number of tokens to replace. Default is 1.0, meaning
        all tokens can be replaced. If this value is float in range
        [0.0, 0.1], it is used as a ratio.
    group_size : int
        When group_size > 1, groups of nearby tokens are replaced all
        in once (each token is still replaced with a replacement).
        Default is 1, meaning individual tokens are replaced.
    """
    def __init__(self,
                 token_pattern=None,  # type: Optional[str]
                 bow=True,            # type: bool
                 random_state=None,
                 replacement='',      # type: str
                 min_replace=1,       # type: Union[int, float]
                 max_replace=1.0,     # type: Union[int, float]
                 group_size=1,        # type: int
                 ):
        # type: (...) -> None
        self.token_pattern = token_pattern or DEFAULT_TOKEN_PATTERN
        self.bow = bow
        self.random_state = random_state
        self.replacement = replacement
        self.min_replace = min_replace
        self.max_replace = max_replace
        self.group_size = group_size
        self.rng_ = check_random_state(self.random_state)

    def sample_near(self, doc, n_samples=1):
        # type: (str, int) -> Tuple[List[str], np.ndarray]
        docs, similarities, mask, text = self.sample_near_with_mask(
            doc=doc, n_samples=n_samples
        )
        return docs, similarities

    def sample_near_with_mask(self,
                              doc,         # type: Union[TokenizedText, str]
                              n_samples=1  # type: int
                              ):
        # type: (...) -> Tuple[List[str], np.ndarray, np.ndarray, TokenizedText]
        if not isinstance(doc, TokenizedText):
            doc = TokenizedText(doc, token_pattern=self.token_pattern)

        gen_samples = partial(generate_samples, doc,
                              n_samples=n_samples,
                              replacement=self.replacement,
                              min_replace=self.min_replace,
                              max_replace=self.max_replace,
                              group_size=self.group_size,
                              random_state=self.rng_)
        docs, similarity, mask = gen_samples(bow=self.bow)
        return docs, similarity, mask, doc


class MaskingTextSamplers(BaseSampler):
    """
    Union of MaskingText samplers, with weights.
    :meth:`sample_near` or :meth:`sample_near_with_mask` generate
    a requested number of samples using all samplers; a probability of
    using a sampler is proportional to its weight.

    All samplers must use the same token_pattern in order for
    :meth:`sample_near_with_mask` to work.

    Create it with a list of {param: value} dicts
    with :class:`MaskingTextSampler` paremeters.
    """
    def __init__(self,
                 sampler_params,      # type: List[Dict[str, Any]]
                 token_pattern=None,  # type: Optional[str]
                 random_state=None,
                 weights=None,        # type: Union[np.ndarray, List[float]]
                 ):
        # type: (...) -> None
        self.random_state = random_state
        self.rng_ = check_random_state(random_state)
        self.token_pattern = token_pattern
        self.samplers = list(map(self._create_sampler, sampler_params))
        if weights is None:
            self.weights = np.ones(len(self.samplers))
        else:
            self.weights = np.array(weights)
        self.weights /= self.weights.sum()

    def _create_sampler(self, extra):
        # type: (Dict) -> MaskingTextSampler
        params = dict(
            token_pattern=self.token_pattern,
            random_state=self.rng_,
        )  # type: Dict[str, Any]
        params.update(extra)
        return MaskingTextSampler(**params)

    def sample_near(self, doc, n_samples=1):
        # type: (str, int) -> Tuple[List[str], np.ndarray]
        assert n_samples >= 1
        all_docs = []  # type: List[str]
        similarities = []
        for sampler, freq in self._sampler_n_samples(n_samples):
            docs, sims = sampler.sample_near(doc, n_samples=freq)
            all_docs.extend(docs)
            similarities.append(sims)
        return all_docs, np.hstack(similarities)

    def sample_near_with_mask(self,
                              doc,         # type: str
                              n_samples=1  # type: int
                              ):
        # type: (...) -> Tuple[List[str], np.ndarray, np.ndarray, TokenizedText]
        assert n_samples >= 1
        assert self.token_pattern is not None
        text = TokenizedText(doc, token_pattern=self.token_pattern)
        all_docs = []  # type: List[str]
        similarities = []
        masks = []
        for sampler, freq in self._sampler_n_samples(n_samples):
            docs, sims, mask, _text = sampler.sample_near_with_mask(text, freq)
            all_docs.extend(docs)
            similarities.append(sims)
            masks.append(mask)
        return all_docs, np.hstack(similarities), vstack(masks), text

    def _sampler_n_samples(self, n_samples):
        """ Return (sampler, n_samplers) tuples """
        sampler_indices = self.rng_.choice(range(len(self.samplers)),
                                           size=n_samples,
                                           replace=True,
                                           p=self.weights)
        return [
            (self.samplers[idx], freq)
            for idx, freq in itemfreq(sampler_indices)
        ]


_BANDWIDTHS = np.hstack([
    [1e-6],                 # for discrete features
    np.logspace(-3, 4, 15)  # general-purpose (0.001 ... 10000) range
])

class _BaseKernelDensitySampler(BaseSampler):
    def __init__(self, kde=None, metric='euclidean', fit_bandwidth=True,
                 bandwidths=_BANDWIDTHS, sigma='bandwidth', n_jobs=1,
                 random_state=None):
        if kde is None:
            kde = KernelDensity(rtol=1e-7, atol=1e-7)
        self.kde = kde
        self.fit_bandwidth = fit_bandwidth
        self.bandwidths = bandwidths
        self.metric = metric
        self.n_jobs = n_jobs
        if not isinstance(sigma, (int, float)):
            allowed = {'bandwidth'}
            if sigma not in allowed:
                raise ValueError("sigma must be either "
                                 "a number or one of {}".format(allowed))
        self.sigma = sigma
        self.random_state = random_state
        self.rng_ = check_random_state(self.random_state)

    def _get_grid(self):
        param_grid = {'bandwidth': self.bandwidths}
        cv = KFold(n_splits=3, shuffle=True, random_state=self.rng_)
        return GridSearchCV(self.kde, param_grid=param_grid, n_jobs=self.n_jobs,
                            cv=cv)

    def _fit_kde(self, kde, X):
        # type: (KernelDensity, np.ndarray) -> Tuple[GridSearchCV, KernelDensity]
        if self.fit_bandwidth:
            grid = self._get_grid()
            grid.fit(X)
            return grid, grid.best_estimator_
        else:
            return None, clone(kde).fit(X)

    def _similarity(self, doc, samples):
        distance = _distances(doc, samples, metric=self.metric)
        return rbf(distance, sigma=self.sigma_)

    def _set_sigma(self, bandwidth):
        if self.sigma == 'bandwidth':
            # Sigma estimation using optimal bandwidth found by KDE.
            self.sigma_ = bandwidth
        else:
            self.sigma_ = self.sigma


class MultivariateKernelDensitySampler(_BaseKernelDensitySampler):
    """
    General-purpose sampler for dense continuous data, based on multivariate
    kernel density estimation.

    The limitation is that a single bandwidth value is used for all dimensions,
    i.e. bandwith matrix is a positive scalar times the identity matrix.
    It is a problem e.g. when features have different variances
    (e.g. some of them are one-hot encoded and other are continuous).
    """
    def fit(self, X, y=None):
        self.grid_, self.kde_ = self._fit_kde(self.kde, X)
        self._set_sigma(self.kde_.bandwidth)
        return self

    def sample_near(self, doc, n_samples=1):
        # XXX: it doesn't sample only near the given document, it
        # samples everywhere
        doc = np.asarray(doc)
        samples = self.kde_.sample(n_samples, random_state=self.rng_)
        return samples, self._similarity(doc, samples)


class UnivariateKernelDensitySampler(_BaseKernelDensitySampler):
    """
    General-purpose sampler for dense continuous data, based on univariate
    kernel density estimation. It estimates a separate probability
    distribution for each input dimension.

    The limitation is that variable interactions are not taken in account.

    Unlike KernelDensitySampler it uses different bandwidths for different
    dimensions; because of that it can handle one-hot encoded features somehow
    (make sure to at least tune the default ``sigma`` parameter).
    Also, at sampling time it replaces only random subsets
    of the features instead of generating totally new examples.
    """
    def fit(self, X, y=None):
        self.kdes_ = []  # type: List[KernelDensity]
        self.grids_ = []  # type: List[GridSearchCV]
        num_features = X.shape[-1]
        for i in range(num_features):
            grid, kde = self._fit_kde(self.kde, X[:, i].reshape(-1, 1))
            self.grids_.append(grid)
            self.kdes_.append(kde)
        self._set_sigma(bandwidth=max(kde.bandwidth for kde in self.kdes_))
        return self

    def sample_near(self, doc, n_samples=1):
        """
        Sample near the document by replacing some of its features
        with values sampled from distribution found by KDE.
        """
        doc = np.asarray(doc)
        num_features = len(self.kdes_)
        sizes = self.rng_.randint(low=1, high=num_features + 1, size=n_samples)
        samples = []
        for size in sizes:
            to_change = self.rng_.choice(num_features, size, replace=False)
            new_doc = doc.copy()
            for i in to_change:
                kde = self.kdes_[i]
                new_doc[i] = kde.sample(random_state=self.rng_).ravel()
            samples.append(new_doc)
        samples = np.asarray(samples)
        return samples, self._similarity(doc, samples)


def _distances(doc, samples, metric):
    doc = doc.reshape(1, -1)
    return pairwise_distances(samples, doc, metric=metric).ravel()
