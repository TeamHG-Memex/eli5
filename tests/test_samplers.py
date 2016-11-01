# -*- coding: utf-8 -*-
from functools import partial

import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import integers, text

from eli5.lime.samplers import (
    MaskingTextSampler,
    UnivariateKernelDensitySampler,
    MultivariateKernelDensitySampler
)
from sklearn.neighbors import KernelDensity


@given(text(), integers(1, 3))
def test_masking_text_sampler_length(text, n_samples):
    for bow in [True, False]:
        sampler = MaskingTextSampler(bow=bow)
        sampler.fit([text])

        samples, distances = sampler.sample_near(text, n_samples=n_samples)
        assert len(samples) == n_samples
        assert distances.shape == (n_samples,)
        assert all(len(s) <= len(text) for s in samples)


def test_masking_text_sampler_bow():
    sampler = MaskingTextSampler(bow=True)
    samples, distances = sampler.sample_near('foo bar bar baz', n_samples=10000)
    assert 'foo bar bar ' in samples
    assert ' bar bar ' in samples
    assert '  bar ' not in samples
    assert ' bar bar baz' in samples
    assert 'foo bar bar baz' not in samples
    assert '   ' in samples
    assert 'foo bar  baz' not in samples
    assert 'foo  bar baz' not in samples


def test_masking_text_sampler():
    sampler = MaskingTextSampler(bow=False)
    samples, distances = sampler.sample_near('foo bar bar baz', n_samples=10000)
    assert 'foo bar bar ' in samples
    assert 'foo  bar baz' in samples
    assert 'foo bar bar baz' not in samples
    assert '   ' in samples


def test_univariate_kde_sampler():
    feat1 = np.random.normal(size=100)
    feat2 = np.random.randint(0, 2, size=100)
    X = np.array([feat1, feat2]).T

    s = UnivariateKernelDensitySampler()
    s.fit(X)

    # second feature is categorical, it should use a small bandwidth
    assert np.isclose(s.kdes_[1].bandwidth, 1e-6)

    # check sampling results
    samples, distances = s.sample_near([0.1, 1], n_samples=1000)

    feat1_sampled = samples[:, 0]
    feat2_sampled = samples[:, 1]

    _isclose = partial(np.isclose, atol=1e-5)

    # feat2 should have both 0 and 1 values, and it should have more 1 values
    # because document has 1 as a second feature
    zeros = _isclose(feat2_sampled, 0)
    ones = _isclose(feat2_sampled, 1)
    assert (zeros | ones).sum() == 1000
    assert 0.5 < feat2_sampled.mean() < 0.9
    assert zeros.sum() > 100
    assert ones.sum() > 500

    # feat1 should be centered around zero
    assert -1 < feat1_sampled.mean() < 1


def test_multivariate_kde_sampler():
    feat1 = np.random.normal(size=500)
    feat2 = feat1 * 2 + np.random.normal(size=500) * 0.01
    X = np.array([feat1, feat2]).T

    s = MultivariateKernelDensitySampler()
    s.fit(X)

    # no extreme bandwidths
    assert 0.01 < s.kde_.bandwidth < 5

    # check sampling results
    X_sampled, distances = s.sample_near(X[0], 1000)
    feat1_sampled = X_sampled[:, 0]
    feat2_sampled = X_sampled[:, 1]

    # feature interaction should be preserved
    assert abs((feat1_sampled * 2 - feat2_sampled).mean()) < 0.05


def test_bad_argument():
    with pytest.raises(ValueError):
        s = MultivariateKernelDensitySampler(sigma='foo')


@pytest.mark.parametrize(['sampler_cls'], [
    [MultivariateKernelDensitySampler],
    [UnivariateKernelDensitySampler]
])
def test_explicit_sigma(sampler_cls):
    X = np.array([[0, 1], [1, 1], [0, 2]])
    s = sampler_cls(sigma=0.5)
    s.fit(X)
    assert s.sigma == 0.5
    assert s.sigma_ == 0.5


def test_sigma_bandwidth():
    s = MultivariateKernelDensitySampler(sigma='bandwidth')
    s.fit([[0, 1], [1, 1], [0, 2]])
    assert s.sigma_ == s.kde_.bandwidth


def test_fit_bandwidth():
    kde = KernelDensity(bandwidth=100, leaf_size=10)
    s = MultivariateKernelDensitySampler(kde=kde, fit_bandwidth=True)
    s.fit([[0, 1], [1, 1], [0, 2]])
    assert s.kde_.bandwidth != kde.bandwidth
    assert s.kde_.leaf_size == kde.leaf_size

    s = MultivariateKernelDensitySampler(kde=kde, fit_bandwidth=False)
    s.fit([[0, 1], [1, 1], [0, 2]])
    assert s.kde_.bandwidth == kde.bandwidth
    assert s.kde_.leaf_size == kde.leaf_size
