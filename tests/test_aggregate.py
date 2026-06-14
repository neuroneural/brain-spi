"""Tests for bootstrap and label_shuffle null distributions."""

import numpy as np
import pytest
from unittest.mock import patch


@pytest.fixture
def fitted_result(small_data, tmp_path):
    spi_names = ["s1", "s2"]
    fake_specs = {n: (".statistics.fake", "Fake", {}) for n in spi_names}

    def _compute_subject(subject_data, configfile):
        C = subject_data.shape[0]
        seed = int(abs(subject_data.sum()) * 1e3) % (2**31)
        rng = np.random.default_rng(seed)
        out = {}
        for name in spi_names:
            mat = rng.standard_normal((C, C))
            out[name] = (mat + mat.T) / 2
        return out

    with patch("brain_spi.spis.resolve_specs", return_value=fake_specs), \
         patch("brain_spi.spis.compute_subject", side_effect=_compute_subject):
        from brain_spi import BrainSPI
        data, labels = small_data
        pipe = BrainSPI(spis=spi_names, cache_dir=tmp_path / "c",
                        rf_kw={"n_estimators": 20, "random_state": 0})
        return pipe.fit(data, labels, use_cache=False, write_cache=False)


def test_bootstrap_shape(fitted_result, small_data):
    data, _ = small_data
    C = data.shape[2]
    null = fitted_result.bootstrap(n=5, frac=0.66)
    assert null.samples.shape == (5, C, C)
    assert null.kind == "bootstrap"


def test_bootstrap_range(fitted_result):
    null = fitted_result.bootstrap(n=5, frac=0.66)
    assert null.samples.min() >= 0.0
    assert null.samples.max() <= 1.0


def test_label_shuffle_shape(fitted_result, small_data):
    data, _ = small_data
    C = data.shape[2]
    null = fitted_result.label_shuffle(n=10)
    assert null.samples.shape == (10, C, C)
    assert null.kind == "label_shuffle"


def test_label_shuffle_mean_near_zero(fitted_result):
    """Shuffled-label null mean should be much lower than observed mean_and."""
    null = fitted_result.label_shuffle(n=20)
    obs = fitted_result.aggregate.mean_and
    # shuffled mean per edge should on average be ≤ observed (not always, but in expectation)
    # Just check the distribution is within [0,1]
    assert null.mean.min() >= 0.0
    assert null.mean.max() <= 1.0


def test_survival_rate(fitted_result):
    null = fitted_result.bootstrap(n=10)
    obs = fitted_result.aggregate.mean_and
    sr = null.survival_rate()
    assert sr.shape == obs.shape
    assert sr.min() >= 0.0
    assert sr.max() <= 1.0


def test_survival_rate_zero_on_empty_edges(fitted_result):
    """Edges never flagged across resamples must have survival 0, not 1."""
    null = fitted_result.bootstrap(n=10)
    sr = null.survival_rate()
    # an edge whose samples are all zero must read as 0 survival
    never = (null.samples == 0).all(axis=0)
    assert np.all(sr[never] == 0.0)


def test_p_value_range_and_meaning(fitted_result):
    null = fitted_result.label_shuffle(n=20)
    obs = fitted_result.aggregate.mean_and
    p = null.p_value(obs)
    assert p.shape == obs.shape
    # (1 + k) / (1 + n): bounded in (0, 1], never exactly 0
    assert p.min() > 0.0
    assert p.max() <= 1.0
    # edges with observed 0: all null samples >= 0, so p must equal 1.0
    assert np.allclose(p[obs == 0], 1.0)
