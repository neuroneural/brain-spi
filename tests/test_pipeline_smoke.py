"""End-to-end smoke test with pyspi mocked out at the spis boundary."""

import numpy as np
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_pyspi():
    """Patch resolve_specs + compute_subject so no real pyspi is needed.

    compute_subject returns deterministic per-subject random matrices keyed on
    the subject data, so caching round-trips are meaningful.
    """
    spi_names = ["fake_spi_A", "fake_spi_B", "fake_spi_C"]
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
        yield spi_names


def test_pipeline_smoke(small_data, mock_pyspi, tmp_path):
    from brain_spi import BrainSPI

    data, labels = small_data
    pipe = BrainSPI(
        spis=mock_pyspi,
        cache_dir=tmp_path / "cache",
    )
    result = pipe.fit(data, labels, use_cache=True, write_cache=True)

    assert result.spis == mock_pyspi
    C = data.shape[2]
    for name in mock_pyspi:
        r = result[name]
        assert r.matrices.shape == (len(labels), C, C)
        assert r.p_thresh.shape == (C, C)
        assert r.rf_mask.shape == (C, C)
        assert r.and_mask.shape == (C, C)
        assert r.and_mask.dtype == bool


def test_aggregate_lazy(small_data, mock_pyspi, tmp_path):
    from brain_spi import BrainSPI

    data, labels = small_data
    pipe = BrainSPI(spis=mock_pyspi, cache_dir=tmp_path / "cache2")
    result = pipe.fit(data, labels, use_cache=False, write_cache=False)

    # aggregate should not exist yet
    assert "aggregate" not in result.__dict__
    agg = result.aggregate
    C = data.shape[2]
    assert agg.mean_and.shape == (C, C)
    assert agg.mean_and.min() >= 0.0
    assert agg.mean_and.max() <= 1.0
    # second access returns same object (cached)
    assert result.aggregate is agg


def test_getitem_missing(small_data, mock_pyspi, tmp_path):
    from brain_spi import BrainSPI

    data, labels = small_data
    pipe = BrainSPI(spis=mock_pyspi, cache_dir=tmp_path / "cache3")
    result = pipe.fit(data, labels, use_cache=False, write_cache=False)

    with pytest.raises(KeyError, match="nonexistent"):
        _ = result["nonexistent"]


def test_cache_roundtrip(small_data, mock_pyspi, tmp_path):
    """Second fit with same data should hit cache and return same matrices."""
    from brain_spi import BrainSPI

    data, labels = small_data
    cache = tmp_path / "cache_rt"
    pipe = BrainSPI(spis=mock_pyspi, cache_dir=cache)

    r1 = pipe.fit(data, labels, use_cache=True, write_cache=True)
    r2 = pipe.fit(data, labels, use_cache=True, write_cache=False)

    for name in mock_pyspi:
        np.testing.assert_array_equal(r1[name].matrices, r2[name].matrices)


def test_pickle_roundtrip(small_data, mock_pyspi, tmp_path):
    from brain_spi import BrainSPI, PipelineResult

    data, labels = small_data
    pipe = BrainSPI(spis=mock_pyspi, cache_dir=tmp_path / "cache_pkl")
    result = pipe.fit(data, labels, use_cache=False, write_cache=False)

    pkl_path = tmp_path / "result.pkl"
    result.to_pickle(pkl_path)
    loaded = PipelineResult.load_pickle(pkl_path)

    for name in mock_pyspi:
        np.testing.assert_array_equal(
            result[name].matrices, loaded[name].matrices
        )


def test_npz_roundtrip(small_data, mock_pyspi, tmp_path):
    import brain_spi
    from brain_spi import BrainSPI

    data, labels = small_data
    pipe = BrainSPI(spis=mock_pyspi, cache_dir=tmp_path / "cache_npz")
    result = pipe.fit(data, labels, use_cache=False, write_cache=False)

    npz_path = tmp_path / "result.npz"
    result.to_npz(npz_path)
    loaded = brain_spi.load_npz(npz_path)

    assert loaded.spis == result.spis
    for name in mock_pyspi:
        for attr in ("matrices", "t_stat", "p_value", "rf_importance"):
            np.testing.assert_array_equal(
                getattr(result[name], attr), getattr(loaded[name], attr)
            )
        for attr in ("p_thresh", "rf_mask", "and_mask"):
            np.testing.assert_array_equal(
                getattr(result[name], attr), getattr(loaded[name], attr)
            )
    # aggregate must reconstruct identically
    np.testing.assert_array_equal(result.aggregate.mean_and, loaded.aggregate.mean_and)


def test_npz_standalone_readable(small_data, mock_pyspi, tmp_path):
    """A user without brain_spi can open the archive with plain numpy.load."""
    from brain_spi import BrainSPI

    data, labels = small_data
    pipe = BrainSPI(spis=mock_pyspi, cache_dir=tmp_path / "cache_npz2")
    result = pipe.fit(data, labels, use_cache=False, write_cache=False)

    npz_path = tmp_path / "result.npz"
    result.to_npz(npz_path)

    z = np.load(npz_path, allow_pickle=False)
    assert "README" in z
    assert [str(x) for x in z["spi_names"]] == mock_pyspi
    np.testing.assert_array_equal(z["spi0_matrices"], result[mock_pyspi[0]].matrices)
    np.testing.assert_array_equal(z["data"], data)
