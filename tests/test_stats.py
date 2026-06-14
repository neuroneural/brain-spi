"""Tests for stats.ttest_edges and stats.rf_features."""

import numpy as np
import pytest
from brain_spi.stats import ttest_edges, rf_features
from brain_spi._utils import tril_indices


def test_ttest_shape(small_data):
    data, labels = small_data
    mats = np.einsum("bti,btj->bij", data, data) / data.shape[1]  # crude corr proxy
    t, p, thresh = ttest_edges(mats, labels)
    C = data.shape[2]
    assert t.shape == (C, C)
    assert p.shape == (C, C)
    assert thresh.shape == (C, C)
    assert thresh.dtype == bool


def test_ttest_symmetric(small_data):
    data, labels = small_data
    mats = np.einsum("bti,btj->bij", data, data) / data.shape[1]
    t, p, _ = ttest_edges(mats, labels)
    np.testing.assert_allclose(t, t.T, atol=1e-10)
    np.testing.assert_allclose(p, p.T, atol=1e-10)


def test_ttest_diagonal_p_one(small_data):
    data, labels = small_data
    mats = np.einsum("bti,btj->bij", data, data) / data.shape[1]
    _, p, _ = ttest_edges(mats, labels)
    C = data.shape[2]
    np.testing.assert_array_equal(np.diag(p), np.ones(C))


def test_ttest_wrong_labels(small_data):
    data, _ = small_data
    mats = np.ones((data.shape[0], data.shape[2], data.shape[2]))
    with pytest.raises(ValueError, match="2 unique labels"):
        ttest_edges(mats, np.zeros(data.shape[0], dtype=int))


def test_rf_features_shape(small_data):
    data, labels = small_data
    mats = np.einsum("bti,btj->bij", data, data) / data.shape[1]
    C = data.shape[2]
    idx = tril_indices(C)
    density = 0.3
    imp, mask = rf_features(mats, labels, density=density)
    assert imp.shape == (C, C)
    assert mask.shape == (C, C)
    assert mask.dtype == bool


def test_rf_features_deterministic(small_data):
    data, labels = small_data
    mats = np.einsum("bti,btj->bij", data, data) / data.shape[1]
    imp1, mask1 = rf_features(mats, labels, density=0.3, rf_kw={"n_estimators": 50, "random_state": 7})
    imp2, mask2 = rf_features(mats, labels, density=0.3, rf_kw={"n_estimators": 50, "random_state": 7})
    np.testing.assert_array_equal(imp1, imp2)
    np.testing.assert_array_equal(mask1, mask2)


def test_matched_density(small_data):
    """RF mask density should match p_thresh density approximately."""
    data, labels = small_data
    mats = np.einsum("bti,btj->bij", data, data) / data.shape[1]
    C = data.shape[2]
    idx = tril_indices(C)
    _, _, p_thresh = ttest_edges(mats, labels)
    density = p_thresh.astype(float)[idx].mean()
    _, rf_mask = rf_features(mats, labels, density=density,
                              rf_kw={"n_estimators": 50, "random_state": 0})
    rf_density = rf_mask.astype(float)[idx].mean()
    # densities should be very close (off by at most 1 edge)
    assert abs(rf_density - density) <= 2 / len(idx[0]) + 1e-6
