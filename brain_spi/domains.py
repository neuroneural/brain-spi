"""ROI domain / network definitions for heatmap guides."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class DomainSpec:
    """Contiguous ranges of ROIs belonging to named networks."""
    names: list[str]
    starts: NDArray  # (D,) int, inclusive start index per domain
    ends: NDArray    # (D,) int, inclusive end index per domain

    @property
    def n_rois(self) -> int:
        return int(self.ends[-1]) + 1

    @property
    def n_domains(self) -> int:
        return len(self.names)

    def tick_positions(self) -> NDArray:
        """Centre of each domain band, for axis ticks."""
        return (self.starts + self.ends) / 2.0

    def boundary_positions(self) -> NDArray:
        """Boundary positions between domains (for grid lines).

        In an imshow image pixel ``i`` spans ``[i-0.5, i+0.5]``, so the line
        between a domain ending at index ``e`` and the next starting at ``e+1``
        sits at ``e+0.5`` — which is exactly the midpoint of the two indices.
        """
        return (self.ends[:-1] + self.starts[1:]) / 2.0


def from_csv(path: str | Path, name_col: str = "Domain") -> DomainSpec:
    """
    Build a DomainSpec from a CSV with one row per ROI.

    The CSV must have a column `name_col` with network labels.  NaNs are
    forward-filled (matching the notebook's `fillna(method='ffill')`).
    """
    df = pd.read_csv(path)
    domains = df[name_col].ffill().to_numpy(dtype=str)

    unique, first_idx = np.unique(domains, return_index=True)
    order = np.argsort(first_idx)
    names_ordered = unique[order].tolist()
    starts_ordered = first_idx[order]

    ends_ordered = np.empty_like(starts_ordered)
    for i, s in enumerate(starts_ordered):
        if i + 1 < len(starts_ordered):
            ends_ordered[i] = starts_ordered[i + 1] - 1
        else:
            ends_ordered[i] = len(domains) - 1

    return DomainSpec(
        names=names_ordered,
        starts=starts_ordered.astype(int),
        ends=ends_ordered.astype(int),
    )


def generic(n_rois: int) -> DomainSpec:
    """Trivial single-domain spec (no visual guides) for unlabelled ROIs."""
    return DomainSpec(
        names=["ROIs"],
        starts=np.array([0], dtype=int),
        ends=np.array([n_rois - 1], dtype=int),
    )
