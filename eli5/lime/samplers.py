# -*- coding: utf-8 -*-
from __future__ import absolute_import
import abc
import six
from typing import Tuple

import numpy as np

from eli5.lime.utils import rbf
from sklearn.base import BaseEstimator, clone
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances
try:
    from sklearn.model_selection import GridSearchCV, KFold
except ImportError:  # scikit-learn < 0.18
    from sklearn.cross_validation import GridSearchCV, KFold

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
        Return (examples, distances) tuple with generated documents
        similar to a given document.
        """
        raise NotImplementedError()

    def fit(self, X=None, y=None):
        return self


class MaskingTextSampler(BaseSampler):
    """
    Sampler for text data. It randomly removes words from text.
    """
    def __init__(self, token_pattern=None, bow=True):
        self.token_pattern = token_pattern or DEFAULT_TOKEN_PATTERN
        self.bow = bow

    def sample_near(self, doc, n_samples=1):
        docs, similarity = generate_samples(doc,
                                            n_samples=n_samples,
                                            bow=self.bow,
                                            token_pattern=self.token_pattern)
        # XXX: should it use RBF kernel as well, instead of raw
        # cosine similarity?
        return list(docs), similarity


_BANDWIDTHS = np.hstack([
    [1e-6],                 # for discrete features
    np.logspace(-3, 4, 15)  # general-purpose (0.001 ... 10000) range
])

class _BaseKernelDensitySampler(BaseSampler):
    def __init__(self, kde=None, metric='euclidean', fit_bandwidth=True,
                 bandwidths=_BANDWIDTHS, sigma='distance', n_jobs=1):
        if kde is None:
            kde = KernelDensity(rtol=1e-7, atol=1e-7)
        self.kde = kde
        self.fit_bandwidth = fit_bandwidth
        self.bandwidths = bandwidths
        self.metric = metric
        self.n_jobs = n_jobs
        if not isinstance(sigma, (int, float)):
            allowed = {'bandwidth', 'distance'}
            if sigma not in allowed:
                raise ValueError("sigma must be either "
                                 "a number or one of {}".format(allowed))
        self.sigma = sigma

    def _get_grid(self):
        param_grid = {'bandwidth': self.bandwidths}
        cv = KFold(n_splits=3, shuffle=True)  # shuffle data by default
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
        if self.sigma == "distance":
            self.sigma_ = abs(distance.max()) / 2
        return rbf(distance, sigma=self.sigma_)

    def _set_sigma(self, bandwidth):
        if self.sigma == 'bandwidth':
            # Sigma estimation using optimal bandwidth found by KDE.
            #
            # If all features are one-hot encoded then bandwidth is
            # tiny, but we need a larger sigma because otherwise similarity
            # for all examples will be close to zero.
            #
            # XXX: how does it work in practice? Is it worth supporting?
            self.sigma_ = max(0.5, bandwidth * 2)
        elif self.sigma == "distance":
            self.sigma_ = None  # depends on sample distances
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
        samples = self.kde_.sample(n_samples)
        return samples, self._similarity(doc, samples)


class UnivariateKernelDensitySampler(_BaseKernelDensitySampler):
    """
    General-purpose sampler for dense continuous data, based on univariate
    kernel density estimation. It estimates a separate probability
    distribution for each input dimension.

    The limitation is that variable interactions are not taken in account.

    Unlike KernelDensitySampler it uses different bandwidths for different
    dimensions; because of that it can handle one-hot encoded features OK.
    Also, at sampling time it replaces only random subsets
    of the features instead of generating totally new examples.
    """
    def fit(self, X, y=None):
        self.kdes_ = []
        self.grids_ = []
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
        sizes = np.random.randint(low=1, high=num_features + 1, size=n_samples)
        samples = []
        for size in sizes:
            to_change = np.random.choice(num_features, size, replace=False)
            new_doc = doc.copy()
            for i in to_change:
                new_doc[i] = self.kdes_[i].sample().ravel()
            samples.append(new_doc)
        samples = np.asarray(samples)
        return samples, self._similarity(doc, samples)


def _distances(doc, samples, metric):
    doc = doc.reshape(1, -1)
    return pairwise_distances(samples, doc, metric=metric).ravel()
