"""AggregateResult, bootstrap (subject resampling), and label-shuffle null distributions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .pipeline import PipelineResult


@dataclass
class AggregateResult:
    """Cross-SPI aggregate masks.  All properties are lazy + cached."""

    _p_thresh_stack: NDArray   # (n_spis, C, C) float
    _rf_mask_stack: NDArray    # (n_spis, C, C) float
    _and_mask_stack: NDArray   # (n_spis, C, C) float

    _mean_and_cache: NDArray | None = field(default=None, init=False, repr=False)
    _mean_p_thresh_cache: NDArray | None = field(default=None, init=False, repr=False)
    _mean_rf_mask_cache: NDArray | None = field(default=None, init=False, repr=False)

    @property
    def mean_and(self) -> NDArray:
        if self._mean_and_cache is None:
            self._mean_and_cache = self._and_mask_stack.mean(axis=0)
        return self._mean_and_cache

    @property
    def mean_p_thresh(self) -> NDArray:
        if self._mean_p_thresh_cache is None:
            self._mean_p_thresh_cache = self._p_thresh_stack.mean(axis=0)
        return self._mean_p_thresh_cache

    @property
    def mean_rf_mask(self) -> NDArray:
        if self._mean_rf_mask_cache is None:
            self._mean_rf_mask_cache = self._rf_mask_stack.mean(axis=0)
        return self._mean_rf_mask_cache

    def plot(self, ax=None, vmax: float | str = "auto", domain_spec=None, save: str | None = None):
        from .viz import plot_aggregate
        return plot_aggregate(self, ax=ax, vmax=vmax, domain_spec=domain_spec, save=save)

    def __repr__(self) -> str:
        n_spis, C, _ = self._and_mask_stack.shape
        return (
            f"<AggregateResult: cross-SPI mean over {n_spis} SPIs, {C}×{C}\n"
            "  .mean_and / .mean_p_thresh / .mean_rf_mask   (C, C) floats in [0, 1]\n"
            "  .plot(domain_spec=..., vmax='auto'|float)>"
        )


@dataclass
class NullDistribution:
    """
    Collection of `mean_and` matrices from bootstrap or label-shuffle runs.

    Attributes
    ----------
    samples    : (n, C, C) array of mean_and matrices from each iteration
    kind       : 'bootstrap' or 'label_shuffle'
    """
    samples: NDArray
    kind: str

    @property
    def mean(self) -> NDArray:
        return self.samples.mean(axis=0)

    @property
    def std(self) -> NDArray:
        return self.samples.std(axis=0)

    def survival_rate(self, threshold: float = 0.0) -> NDArray:
        """
        Per-edge fraction of resamples in which the edge is flagged.

        This is the **bootstrap reproducibility** measure: the proportion of
        resamples whose ``mean_and`` exceeds ``threshold`` (default 0, i.e. the
        edge was flagged by at least one SPI). Edges that are robustly
        discriminating survive most resamples (→ near 1); edges that are noise
        are rarely flagged (→ near 0).

        Returns (C, C) in [0, 1].

        Note
        ----
        Do **not** pass the observed matrix here — survival is about how often
        each edge reappears under resampling, independent of its observed value.
        For a permutation significance test against a label-shuffle null, use
        :meth:`p_value` instead.
        """
        return (self.samples > threshold).mean(axis=0)

    def p_value(self, observed: NDArray, tail: str = "greater") -> NDArray:
        """
        Per-edge permutation p-value against this null distribution.

        ``p = (1 + #{samples >= observed}) / (1 + n)``  (the +1 avoids p=0 and
        is the standard small-sample correction). Use with a ``label_shuffle``
        null: **low** p means the observed edge is unlikely under the null.

        Returns (C, C) in (0, 1].
        """
        n = self.samples.shape[0]
        if tail == "greater":
            k = (self.samples >= observed[None]).sum(axis=0)
        elif tail == "less":
            k = (self.samples <= observed[None]).sum(axis=0)
        else:
            raise ValueError("tail must be 'greater' or 'less'")
        return (1 + k) / (1 + n)

    def __repr__(self) -> str:
        n, C, _ = self.samples.shape
        return (
            f"<NullDistribution ({self.kind}): {n} samples, {C}×{C}\n"
            "  .samples (n, C, C)  .mean  .std\n"
            "  .survival_rate(threshold=0)   bootstrap: how often each edge is flagged\n"
            "  .p_value(observed)            label_shuffle: permutation p-value>"
        )


def _run_analysis_on_subset(
    data: NDArray,
    labels: NDArray,
    spi_matrices: dict[str, NDArray],
    rf_kw: dict,
) -> NDArray:
    """Run per-SPI stats on a subject subset and return the mean_and matrix."""
    from .stats import ttest_edges, rf_features
    from ._utils import tril_indices

    and_masks = []
    C = data.shape[2]
    for name, mats in spi_matrices.items():
        try:
            _, _, p_thresh = ttest_edges(mats, labels)
            density = p_thresh.astype(float)[np.tril_indices(C, k=-1)].mean()
            _, rf_mask = rf_features(mats, labels, density=density, rf_kw=rf_kw)
            and_masks.append((p_thresh & rf_mask).astype(float))
        except Exception:
            continue

    if not and_masks:
        return np.zeros((C, C))
    return np.stack(and_masks, axis=0).mean(axis=0)


def bootstrap(
    result: "PipelineResult",
    n: int = 20,
    frac: float = 0.66,
    rng: int | np.random.Generator = 0,
) -> NullDistribution:
    """
    Subject-resampling robustness check.

    For each of `n` iterations, draw `frac` of subjects (without replacement),
    rerun per-SPI group analysis, and compute mean_and.

    Returns a NullDistribution with samples of shape (n, C, C).
    """
    rng = np.random.default_rng(rng)
    B = result.data.shape[0]
    k = max(2, int(round(frac * B)))

    samples = []
    for _ in range(n):
        idx = rng.choice(B, size=k, replace=False)
        sub_data = result.data[idx]
        sub_labels = result.labels[idx]
        sub_mats = {name: result[name].matrices[idx] for name in result.spis}
        mean_and = _run_analysis_on_subset(sub_data, sub_labels, sub_mats, result._rf_kw)
        samples.append(mean_and)

    return NullDistribution(samples=np.stack(samples, axis=0), kind="bootstrap")


def label_shuffle(
    result: "PipelineResult",
    n: int = 100,
    rng: int | np.random.Generator = 0,
) -> NullDistribution:
    """
    Permutation null: shuffle labels `n` times and rerun stats.

    Returns a NullDistribution whose mean should be near zero if the observed
    mean_and is not a chance result.
    """
    rng = np.random.default_rng(rng)
    B = result.data.shape[0]
    spi_mats = {name: result[name].matrices for name in result.spis}

    samples = []
    for _ in range(n):
        shuffled_labels = rng.permutation(result.labels)
        mean_and = _run_analysis_on_subset(result.data, shuffled_labels, spi_mats, result._rf_kw)
        samples.append(mean_and)

    return NullDistribution(samples=np.stack(samples, axis=0), kind="label_shuffle")
