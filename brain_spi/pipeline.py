"""BrainSPI, PipelineResult, PerSPIResult — core pipeline assembly."""

from __future__ import annotations

import pickle
import warnings
from functools import cached_property
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .config import DEFAULT_RF_KW, DEFAULT_T_KW, CACHE_DIR_DEFAULT
from ._utils import tril_indices


# ---------------------------------------------------------------------------
# PerSPIResult
# ---------------------------------------------------------------------------

class PerSPIResult:
    """Artifacts for a single SPI."""

    def __init__(
        self,
        name: str,
        matrices: NDArray,       # (B, C, C)
        labels: NDArray,         # (B,)
        group_values: NDArray,   # sorted unique label values
        t_stat: NDArray,
        p_value: NDArray,
        p_thresh: NDArray,
        rf_importance: NDArray,
        rf_mask: NDArray,
    ):
        self.name = name
        self.matrices = matrices
        self._labels = labels
        self._group_values = group_values
        self.t_stat = t_stat
        self.p_value = p_value
        self.p_thresh = p_thresh
        self.rf_importance = rf_importance
        self.rf_mask = rf_mask

    @property
    def and_mask(self) -> NDArray:
        return self.p_thresh & self.rf_mask

    def mean_matrix(self, group: int | None = None) -> NDArray:
        """
        Return mean (C, C) matrix.

        Parameters
        ----------
        group : 0 or 1 — if given, average within that group only.
        """
        if group is None:
            return self.matrices.mean(axis=0)
        val = self._group_values[group]
        mask = self._labels == val
        return self.matrices[mask].mean(axis=0)

    def plot_triptych(self, ax=None, domain_spec=None, save=None):
        from .viz import plot_triptych
        return plot_triptych(self, ax=ax, domain_spec=domain_spec, save=save, spi_name=self.name)

    def plot_mean(self, ax=None, by_group=False, group_names=("Group 0", "Group 1"),
                  domain_spec=None):
        from .viz import plot_mean
        return plot_mean(self, ax=ax, by_group=by_group, group_names=group_names,
                         domain_spec=domain_spec, spi_name=self.name)

    def __repr__(self) -> str:
        B, C, _ = self.matrices.shape
        n_edges = C * (C - 1) // 2
        return (
            f"<PerSPIResult '{self.name}': {B} subjects, {C}×{C} matrices "
            f"({n_edges} edges)\n"
            f"  sig-p: {int(self.p_thresh.sum() // 2)}  "
            f"rf: {int(self.rf_mask.sum() // 2)}  "
            f"and: {int(self.and_mask.sum() // 2)} edges  |  call .help() for the API>"
        )

    def help(self) -> None:
        """Print a summary of the available attributes and methods."""
        print(
            f"PerSPIResult '{self.name}' — artifacts for one SPI\n"
            "\n"
            "  Data\n"
            "    .matrices            (B, C, C) per-subject connectivity\n"
            "    .mean_matrix()       (C, C) mean across subjects\n"
            "    .mean_matrix(group=0)  mean within one group (0 or 1)\n"
            "\n"
            "  Statistics (all (C, C))\n"
            "    .t_stat, .p_value    Welch t-test t-statistic / p-value\n"
            "    .p_thresh            bool mask, Bonferroni-significant edges\n"
            "    .rf_importance       Random-Forest edge importance\n"
            "    .rf_mask             bool mask, top RF edges (matched density)\n"
            "    .and_mask            p_thresh & rf_mask\n"
            "\n"
            "  Plots\n"
            "    .plot_triptych(domain_spec=..., save=...)   sig-p / RF / AND\n"
            "    .plot_mean(by_group=True, domain_spec=...)  mean connectivity"
        )

    # Calling the object (result['spi']()) just shows the help.
    def __call__(self) -> None:
        self.help()


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

class PipelineResult:
    """
    Container returned by BrainSPI.fit().

    Access per-SPI results via result['spi_name'].
    Access cross-SPI aggregate via result.aggregate (lazy, cached).
    """

    def __init__(
        self,
        spi_results: dict[str, PerSPIResult],
        data: NDArray,
        labels: NDArray,
        rf_kw: dict,
    ):
        self._spi_results = spi_results
        self.data = data
        self.labels = labels
        self._rf_kw = rf_kw

    def __getitem__(self, spi_name: str) -> PerSPIResult:
        if spi_name not in self._spi_results:
            available = ", ".join(self._spi_results)
            raise KeyError(f"SPI '{spi_name}' not found. Available: {available}")
        return self._spi_results[spi_name]

    @property
    def spis(self) -> list[str]:
        return list(self._spi_results.keys())

    @cached_property
    def aggregate(self):
        from .aggregate import AggregateResult
        names = self.spis
        p_stack = np.stack([self[n].p_thresh.astype(float) for n in names], axis=0)
        rf_stack = np.stack([self[n].rf_mask.astype(float) for n in names], axis=0)
        and_stack = np.stack([self[n].and_mask.astype(float) for n in names], axis=0)
        return AggregateResult(
            _p_thresh_stack=p_stack,
            _rf_mask_stack=rf_stack,
            _and_mask_stack=and_stack,
        )

    def bootstrap(self, n: int = 20, frac: float = 0.66, rng: int | np.random.Generator = 0):
        from .aggregate import bootstrap
        return bootstrap(self, n=n, frac=frac, rng=rng)

    def label_shuffle(self, n: int = 100, rng: int | np.random.Generator = 0):
        from .aggregate import label_shuffle
        return label_shuffle(self, n=n, rng=rng)

    def to_pickle(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickle(path: str | Path) -> "PipelineResult":
        with open(path, "rb") as f:
            return pickle.load(f)

    def to_npz(self, path: str | Path) -> None:
        """
        Save all data to a single portable ``.npz`` archive.

        Unlike :meth:`to_pickle`, the result is a flat collection of plain numpy
        arrays that anyone can open with ``numpy.load(path)`` *without* having
        ``brain_spi`` installed. The archive includes a ``README`` entry
        describing the key layout. Round-trips losslessly via :meth:`from_npz`.
        """
        import json

        names = self.spis
        groups = np.unique(self.labels)
        readme = (
            "brain_spi PipelineResult export (format brain_spi-npz-v1).\n"
            "Open with: z = numpy.load(path, allow_pickle=False)\n"
            "\n"
            "Top-level keys:\n"
            "  spi_names      (n_spis,) str   — SPI identifiers, in order\n"
            "  data           (B, T, C)       — input time series\n"
            "  labels         (B,)            — group labels\n"
            "  group_values   (2,)            — sorted unique label values\n"
            "  rf_kw_json     ()  str         — RandomForest kwargs (JSON)\n"
            "\n"
            "Per-SPI keys (i = index into spi_names):\n"
            "  spi{i}_matrices       (B, C, C) per-subject connectivity\n"
            "  spi{i}_t_stat         (C, C) Welch t-statistic\n"
            "  spi{i}_p_value        (C, C) t-test p-value\n"
            "  spi{i}_p_thresh       (C, C) bool, Bonferroni-significant edges\n"
            "  spi{i}_rf_importance  (C, C) Random-Forest edge importance\n"
            "  spi{i}_rf_mask        (C, C) bool, top RF edges\n"
            "AND mask = p_thresh & rf_mask. Aggregate = mean over SPIs."
        )

        arrays: dict[str, np.ndarray] = {
            "format_version": np.array("brain_spi-npz-v1"),
            "README": np.array(readme),
            "spi_names": np.array(names),
            "data": np.asarray(self.data),
            "labels": np.asarray(self.labels),
            "group_values": np.asarray(groups),
            "rf_kw_json": np.array(json.dumps(self._rf_kw)),
        }
        for i, name in enumerate(names):
            r = self[name]
            arrays[f"spi{i}_matrices"] = r.matrices
            arrays[f"spi{i}_t_stat"] = r.t_stat
            arrays[f"spi{i}_p_value"] = r.p_value
            arrays[f"spi{i}_p_thresh"] = r.p_thresh
            arrays[f"spi{i}_rf_importance"] = r.rf_importance
            arrays[f"spi{i}_rf_mask"] = r.rf_mask

        np.savez_compressed(path, **arrays)

    @staticmethod
    def from_npz(path: str | Path) -> "PipelineResult":
        """Reconstruct a :class:`PipelineResult` from a :meth:`to_npz` archive."""
        import json

        z = np.load(path, allow_pickle=False)
        names = [str(x) for x in z["spi_names"]]
        data = z["data"]
        labels = z["labels"]
        groups = z["group_values"] if "group_values" in z else np.unique(labels)
        rf_kw = json.loads(str(z["rf_kw_json"])) if "rf_kw_json" in z else {}

        spi_results: dict[str, PerSPIResult] = {}
        for i, name in enumerate(names):
            spi_results[name] = PerSPIResult(
                name=name,
                matrices=z[f"spi{i}_matrices"],
                labels=labels,
                group_values=groups,
                t_stat=z[f"spi{i}_t_stat"],
                p_value=z[f"spi{i}_p_value"],
                p_thresh=z[f"spi{i}_p_thresh"].astype(bool),
                rf_importance=z[f"spi{i}_rf_importance"],
                rf_mask=z[f"spi{i}_rf_mask"].astype(bool),
            )
        return PipelineResult(spi_results=spi_results, data=data, labels=labels, rf_kw=rf_kw)

    def __repr__(self) -> str:
        B = self.data.shape[0]
        C = self.data.shape[2]
        agg_state = "computed" if "aggregate" in self.__dict__ else "lazy"
        spis = self.spis
        shown = ", ".join(spis[:4]) + (", …" if len(spis) > 4 else "")
        return (
            f"<PipelineResult: {len(spis)} SPIs × {B} subjects, {C}×{C} matrices\n"
            f"  SPIs: {shown}\n"
            f"  aggregate: {agg_state}  |  call .help() for the API>"
        )

    def help(self) -> None:
        """Print a summary of the available attributes and methods."""
        print(
            f"PipelineResult — {len(self.spis)} SPIs × {self.data.shape[0]} subjects\n"
            "\n"
            "  Per-SPI\n"
            "    result['<spi>']        -> PerSPIResult (call its .help() too)\n"
            "    result.spis            list of SPI names\n"
            "\n"
            "  Cross-SPI aggregate (lazy, cached on first access)\n"
            "    result.aggregate       -> AggregateResult\n"
            "    result.aggregate.mean_and / .plot()\n"
            "\n"
            "  Robustness / null distributions\n"
            "    result.bootstrap(n=20, frac=0.66)   subject resampling\n"
            "    result.label_shuffle(n=100)         permutation null\n"
            "\n"
            "  Save / load (recommended after a long fit)\n"
            "    result.to_pickle('result.pkl')\n"
            "    PipelineResult.load_pickle('result.pkl')\n"
            "\n"
            f"  SPIs: {', '.join(self.spis)}"
        )

    # Calling the object (result()) just shows the help.
    def __call__(self) -> None:
        self.help()


# ---------------------------------------------------------------------------
# BrainSPI
# ---------------------------------------------------------------------------

class BrainSPI:
    """
    Main entry point for the SPI connectivity-analysis pipeline.

    Parameters
    ----------
    spis       : SPI selection — None for default, 'spis_all' for full set,
                 path string, or explicit list of names.
    cache_dir  : override default cache directory (~/.cache/brain_spi).
    rf_kw      : override default RandomForest kwargs.
    group_names: human-readable names for the two groups.
    """

    def __init__(
        self,
        spis: str | list[str] | None = None,
        cache_dir: str | Path | None = None,
        rf_kw: dict | None = None,
        group_names: tuple[str, str] = ("Group 0", "Group 1"),
    ):
        self.spis = spis
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rf_kw = {**DEFAULT_RF_KW, **(rf_kw or {})}
        self.group_names = group_names

    def fit(
        self,
        data: NDArray,
        labels: NDArray,
        use_cache: bool = True,
        write_cache: bool = True,
        progress: bool = True,
    ) -> PipelineResult:
        """
        Run the full pipeline.

        Parameters
        ----------
        data   : (B, T, C)
        labels : (B,) with exactly two unique values

        Returns
        -------
        PipelineResult
        """
        from .spis import compute_all
        from .stats import ttest_edges, rf_features

        data = np.asarray(data, dtype=float)
        labels = np.asarray(labels)

        groups = np.unique(labels)
        if len(groups) != 2:
            raise ValueError(f"Expected exactly 2 labels, got {len(groups)}")

        B, T, C = data.shape
        idx = tril_indices(C)
        n_edges = len(idx[0])

        # Stage 1 — SPI computation (with cache)
        spi_matrices = compute_all(
            data,
            spis=self.spis,
            cache_dir=self.cache_dir,
            use_cache=use_cache,
            write_cache=write_cache,
            progress=progress,
        )

        # Stage 2 — per-SPI group analysis
        spi_results: dict[str, PerSPIResult] = {}
        for name, mats in spi_matrices.items():
            t_stat, p_value, p_thresh = ttest_edges(mats, labels)
            density = p_thresh.astype(float)[idx].mean()
            rf_importance, rf_mask = rf_features(mats, labels, density=density, rf_kw=self.rf_kw)
            spi_results[name] = PerSPIResult(
                name=name,
                matrices=mats,
                labels=labels,
                group_values=groups,
                t_stat=t_stat,
                p_value=p_value,
                p_thresh=p_thresh,
                rf_importance=rf_importance,
                rf_mask=rf_mask,
            )

        return PipelineResult(
            spi_results=spi_results,
            data=data,
            labels=labels,
            rf_kw=self.rf_kw,
        )
