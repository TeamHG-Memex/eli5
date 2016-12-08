# -*- coding: utf-8 -*-
from __future__ import absolute_import
import abc
import six
import math
from typing import List, Tuple
from functools import partial

import numpy as np

from eli5.lime.utils import rbf
from sklearn.base import BaseEstimator, clone
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils import check_random_state

from .textutils import generate_samples, DEFAULT_TOKEN_PATTERN


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
    Sampler for text data. It randomly removes tokens from text.

    Parameters
    ----------
    token_pattern : str, optional
        Regexp for token matching
    bow : bool or float, optional
        Sampler could either remove all instances of a given token
        (bag of words sampling) or remove just a sing token.
        ``bow`` argument is a ratio of samples which had all instances
        of a token removed. True means 1.0, False means 0.0.
    random_state : integer or numpy.random.RandomState, optional
        random state
    """
    def __init__(self, token_pattern=None, bow=1.0, random_state=None):
        if not (0 <= bow <= 1.0):
            raise ValueError("bow argument is out of "
                             "[0, 1] range: {}".format(bow))
        self.token_pattern = token_pattern or DEFAULT_TOKEN_PATTERN
        self.bow = float(bow)
        self.random_state = random_state
        self.rng_ = check_random_state(self.random_state)

    def sample_near(self, doc, n_samples=1):
        # type: (str, int) -> Tuple[List[str], np.ndarray]
        n_bow = int(math.ceil(self.bow * n_samples))
        n_not_bow = int(math.floor((1 - self.bow) * n_samples))
        gen_samples = partial(generate_samples,
                              doc,
                              token_pattern=self.token_pattern,
                              random_state=self.rng_)

        all_docs = []
        similarities = []
        if n_bow:
            docs, similarity = gen_samples(bow=True, n_samples=n_bow)
            all_docs.extend(docs)
            similarities.append(similarity)
        if n_not_bow:
            docs, similarity = gen_samples(bow=False, n_samples=n_not_bow)
            all_docs.extend(docs)
            similarities.append(similarity)

        # XXX: should it use RBF kernel as well, instead of raw
        # cosine similarity?
        if similarities:
            similarities = np.hstack(similarities)
        else:
            similarities = np.array(similarities)
        return list(all_docs), similarities


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
