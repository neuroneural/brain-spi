"""Shared synthetic fixtures."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def small_data(rng):
    """(B=10, T=50, C=8) synthetic data with two groups."""
    B, T, C = 10, 50, 8
    data = rng.standard_normal((B, T, C))
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return data, labels
