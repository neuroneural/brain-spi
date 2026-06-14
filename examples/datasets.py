"""Download and load the ABIDE1 and COBRE fMRI ICA datasets.

This is a helper for the example notebooks — it is **not** part of the
``brain_spi`` package, and the datasets themselves are not shipped with it.
Both are downloaded from public mirrors on first use.

Each dataset is 100-component ICA time series reshaped to 53 validated
intrinsic connectivity networks (ICNs) over 140 TRs, returned in brain_spi's
``(B, T, C)`` layout together with binary group labels.

    from datasets import download, load

    download("cobre")                  # fetch files (once)
    data, labels, names = load("cobre")  # (B, 140, 53), (B,), ('control', 'SZ')

Adapted from https://github.com/paavalipopov/intro-dl-project
(scripts/download_datasets.py and src/data.py).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

DATA_ROOT = Path(__file__).parent / "datasets_data"

_MILC = "https://raw.githubusercontent.com/UsmanMahmood27/MILC/master"
_INTRO = "https://raw.githubusercontent.com/paavalipopov/intro-dl-project/main"

# Per-dataset config: remote files, HDF5 key, and human-readable group names.
# Labels are stored 1-indexed; we subtract 1, so the lower value is group 0.
DATASETS: dict[str, dict] = {
    "abide": {
        "files": {
            "data": (f"{_MILC}/Data/ABIDE1_AllData.h5", "ABIDE1_AllData.h5"),
            "labels": (f"{_MILC}/IndicesAndLabels/labels_ABIDE1.csv", "labels_ABIDE1.csv"),
            "ica": (f"{_INTRO}/ICA_correct_order.csv", "ICA_correct_order.csv"),
            "domains": (f"{_INTRO}/ICN_coordinates.csv", "ICN_coordinates.csv"),
        },
        "h5_key": "ABIDE1_dataset",
        "group_names": ("control", "autism"),
    },
    "cobre": {
        "files": {
            "data": (f"{_MILC}/Data/COBRE_AllData.h5", "COBRE_AllData.h5"),
            "labels": (f"{_MILC}/IndicesAndLabels/labels_COBRE.csv", "labels_COBRE.csv"),
            "ica": (f"{_INTRO}/ICA_correct_order.csv", "ICA_correct_order.csv"),
            "domains": (f"{_INTRO}/ICN_coordinates.csv", "ICN_coordinates.csv"),
        },
        "h5_key": "COBRE_dataset",
        "group_names": ("control", "schizophrenia"),
    },
}

_N_COMPONENTS = 100
_N_TIMEPOINTS = 140


def download(dataset: str, root: Path | str = DATA_ROOT, force: bool = False) -> Path:
    """
    Download the raw files for ``dataset`` ('abide' or 'cobre') into ``root``.

    Returns the directory containing the files. Existing files are skipped
    unless ``force=True``.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASETS)}.")

    out_dir = Path(root) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for url, fname in DATASETS[dataset]["files"].values():
        dest = out_dir / fname
        if dest.exists() and not force:
            print(f"[skip] {dest.name} already present")
            continue
        print(f"[get ] {url}  ->  {dest}")
        urllib.request.urlretrieve(url, dest)  # noqa: S310 — trusted public mirror

    return out_dir


def load(
    dataset: str,
    root: Path | str = DATA_ROOT,
    zscore: bool = True,
    drop_nan: bool = True,
) -> tuple[NDArray, NDArray, tuple[str, str]]:
    """
    Load ``dataset`` into ``(data, labels, group_names)``.

    Parameters
    ----------
    dataset  : 'abide' or 'cobre'.
    root     : directory the files were downloaded to.
    zscore   : z-score each component along time (per subject).
    drop_nan : drop any subject containing NaNs.

    Returns
    -------
    data        : (B, T=140, C=53) float
    labels      : (B,) int, zero-indexed binary
    group_names : (str, str) names for label 0 and 1
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASETS)}.")

    cfg = DATASETS[dataset]
    data_dir = Path(root) / dataset
    files = cfg["files"]

    data_path = data_dir / files["data"][1]
    labels_path = data_dir / files["labels"][1]
    ica_path = data_dir / files["ica"][1]

    missing = [p for p in (data_path, labels_path, ica_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files for '{dataset}': {[p.name for p in missing]}. "
            f"Run download('{dataset}') first."
        )

    # Features: (B, 14000) -> (B, 100, 140)
    with h5py.File(data_path, "r") as hf:
        raw = np.array(hf.get(cfg["h5_key"]))
    B = raw.shape[0]
    data = raw.reshape(B, _N_COMPONENTS, -1)  # (B, 100, 140)

    # Filter to the 53 validated ICN components (1-indexed in the csv).
    idx = pd.read_csv(ica_path, header=None)[0].to_numpy() - 1
    data = data[:, idx, :]  # (B, 53, 140)

    # -> (B, T, C) layout expected by brain_spi
    data = np.swapaxes(data, 1, 2).astype(float)  # (B, 140, 53)

    if zscore:
        data = stats.zscore(data, axis=1)  # along time

    # Labels: stored 1-indexed -> zero-index.
    labels = pd.read_csv(labels_path, header=None).to_numpy().flatten().astype(int) - 1

    if drop_nan:
        keep = ~np.isnan(data).any(axis=(1, 2))
        if not keep.all():
            print(f"[load] dropping {int((~keep).sum())} subject(s) with NaNs")
        data, labels = data[keep], labels[keep]

    return data, labels, cfg["group_names"]


def domain_spec(dataset: str, root: Path | str = DATA_ROOT):
    """
    Build a brain_spi ``DomainSpec`` for the 53 ICN components from the
    downloaded ``ICN_coordinates.csv`` (run :func:`download` first).

    The component order matches the ``ICA_correct_order`` reindexing applied in
    :func:`load`, so the domain bands line up with the connectivity matrices.
    """
    import brain_spi.domains as domains

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASETS)}.")
    csv = Path(root) / dataset / DATASETS[dataset]["files"]["domains"][1]
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv.name} not found for '{dataset}'. Run download('{dataset}') first."
        )
    return domains.from_csv(csv, name_col="Domain")
