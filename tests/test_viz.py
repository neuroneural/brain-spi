"""Smoke tests that figures render without error."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from brain_spi.aggregate import AggregateResult
from brain_spi.pipeline import PerSPIResult
from brain_spi.viz import plot_heatmap, plot_triptych, plot_aggregate, plot_mean


@pytest.fixture
def dummy_spi_result():
    rng = np.random.default_rng(0)
    C = 8
    B = 6
    mats = rng.standard_normal((B, C, C))
    labels = np.array([0, 0, 0, 1, 1, 1])
    groups = np.array([0, 1])
    t = rng.standard_normal((C, C))
    p = np.abs(rng.standard_normal((C, C))) * 0.1
    p_thresh = p < 0.05
    rf_imp = np.abs(rng.standard_normal((C, C)))
    rf_mask = rf_imp > rf_imp.mean()
    return PerSPIResult(
        name="test_spi",
        matrices=mats,
        labels=labels,
        group_values=groups,
        t_stat=t,
        p_value=p,
        p_thresh=p_thresh,
        rf_importance=rf_imp,
        rf_mask=rf_mask,
    )


@pytest.fixture
def dummy_agg():
    rng = np.random.default_rng(1)
    C, n = 8, 3
    s = rng.uniform(0, 1, (n, C, C))
    return AggregateResult(_p_thresh_stack=s, _rf_mask_stack=s, _and_mask_stack=s)


def test_plot_heatmap_smoke(dummy_spi_result):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    im = plot_heatmap(dummy_spi_result.mean_matrix(), ax=ax)
    assert im is not None
    plt.close(fig)


def test_plot_triptych_smoke(dummy_spi_result):
    import matplotlib.pyplot as plt
    fig = plot_triptych(dummy_spi_result)
    assert fig is not None
    plt.close(fig)


def test_plot_aggregate_smoke(dummy_agg):
    import matplotlib.pyplot as plt
    fig = plot_aggregate(dummy_agg)
    assert fig is not None
    plt.close(fig)


def test_plot_mean_smoke(dummy_spi_result):
    import matplotlib.pyplot as plt
    fig = plot_mean(dummy_spi_result)
    assert fig is not None
    plt.close(fig)


def test_plot_mean_by_group_smoke(dummy_spi_result):
    import matplotlib.pyplot as plt
    fig = plot_mean(dummy_spi_result, by_group=True)
    assert fig is not None
    plt.close(fig)
