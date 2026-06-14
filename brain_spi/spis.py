"""pyspi config building, per-subject SPI computation, caching, and validation.

Key design point
----------------
pyspi's *default* configuration imports every statistics module — including
``.statistics.causal``, which depends on ``pyEDM``/``pkg_resources`` and breaks
under setuptools >= 81.  Passing ``configfile=None`` to a pyspi ``Calculator``
therefore (a) computes all ~250 SPIs regardless of what the user asked for, and
(b) crashes during module import.

To avoid both problems we never let pyspi load its default config.  Instead we:

1. *Discover* the mapping ``identifier -> (module, class, params)`` by importing
   only the modules that load successfully (broken ones are skipped with a
   warning).  This is cached to disk.
2. Resolve the user's SPI selection to a set of concrete specs.
3. Write a *pruned* pyspi config containing exactly those SPIs, so only the
   needed modules are imported and only the requested SPIs are computed.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import logging
import os
import pickle
import tempfile
import warnings
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from ._utils import hash_array, hash_string

try:
    from pyspi.calculator import Calculator
except ImportError:  # pyspi optional at import time; tests mock this name
    Calculator = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_PKG_CONFIGS = Path(__file__).parent / "configs"


def _get_tqdm():
    """Return tqdm.auto.tqdm, or None if tqdm is unavailable."""
    try:
        from tqdm.auto import tqdm
        return tqdm
    except ImportError:
        return None

# A spec is (module_name, class_name, params_dict)
Spec = tuple


# ---------------------------------------------------------------------------
# SPI discovery (identifier -> spec), cached on disk
# ---------------------------------------------------------------------------

def _pyspi_default_config_path() -> Path:
    import pyspi
    return Path(pyspi.__file__).parent / "config.yaml"


def _discover_specs(config_path: Path, cache_dir: Path) -> dict[str, Spec]:
    """
    Build ``{identifier: (module, class, params)}`` for every SPI in
    ``config_path`` that can actually be instantiated.

    Modules that fail to import (e.g. ``.statistics.causal`` under recent
    setuptools) are skipped with a warning.  Result is cached to disk keyed on
    the config file's path + mtime.
    """
    if Calculator is None:
        raise ImportError("pyspi is required for SPI computation. Install it first.")

    config_path = Path(config_path)
    mtime = config_path.stat().st_mtime
    key = hash_string(f"{config_path}:{mtime}")
    cache_file = cache_dir / f"spi_map_{key[:16]}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    with open(config_path) as f:
        yf = yaml.load(f, Loader=yaml.FullLoader)

    mapping: dict[str, Spec] = {}
    broken: list[str] = []

    # pyspi modules are noisy on import; swallow their stdout.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        for module_name in yf:
            try:
                module = importlib.import_module(module_name, "pyspi")
            except Exception as e:  # noqa: BLE001 — want to skip *any* import failure
                broken.append(f"{module_name} ({type(e).__name__})")
                continue
            for fcn in yf[module_name]:
                configs = yf[module_name][fcn].get("configs") or [{}]
                for params in configs:
                    params = params or {}
                    try:
                        spi = getattr(module, fcn)(**params)
                    except TypeError:
                        try:
                            spi = getattr(module, fcn)()
                        except Exception:  # noqa: BLE001
                            continue
                    except Exception:  # noqa: BLE001
                        continue
                    mapping[spi.identifier] = (module_name, fcn, dict(params))

    if broken:
        warnings.warn(
            "Some pyspi modules could not be imported and their SPIs are "
            f"unavailable: {', '.join(broken)}. SPIs in these modules cannot "
            "be computed in this environment.",
            stacklevel=2,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(mapping, f)
    except Exception:
        pass

    return mapping


# ---------------------------------------------------------------------------
# Selection resolution: spis -> {identifier: spec}
# ---------------------------------------------------------------------------

def _is_pyspi_format(doc: dict) -> bool:
    """A pyspi config has top-level module keys like '.statistics.basic'."""
    return any(str(k).startswith(".statistics") for k in doc)


def resolve_specs(
    spis: str | list[str] | None,
    cache_dir: Path,
) -> dict[str, Spec]:
    """
    Resolve a user SPI selection into ``{identifier: (module, class, params)}``.

    Accepts:
      * ``None`` / ``'spis_default'`` — the curated 9-SPI default list.
      * an inline list of identifier strings.
      * a packaged stem (``'spis_all'``) or a path to a YAML file — either a
        flat ``spis: [...]`` list or a full pyspi-format config.
    """
    full_map = _discover_specs(_pyspi_default_config_path(), cache_dir)

    def _subset(names: list[str]) -> dict[str, Spec]:
        out: dict[str, Spec] = {}
        missing: list[str] = []
        for n in names:
            if n in full_map:
                out[n] = full_map[n]
            else:
                missing.append(n)
        if missing:
            warnings.warn(
                f"Requested SPIs not available (unknown or in a module that "
                f"failed to import): {missing}. They will be skipped.",
                stacklevel=2,
            )
        return out

    if spis is None or spis == "spis_default":
        from .config import DEFAULT_SPIS
        return _subset(DEFAULT_SPIS)

    if isinstance(spis, list):
        return _subset(list(spis))

    if isinstance(spis, str):
        candidate = _PKG_CONFIGS / f"{spis}.yaml"
        path = candidate if candidate.exists() else Path(spis)
        if not path.exists():
            raise FileNotFoundError(f"SPI config not found: {spis}")
        with open(path) as f:
            doc = yaml.safe_load(f)
        if _is_pyspi_format(doc):
            # discover identifiers produced by this specific pyspi config
            return _discover_specs(path, cache_dir)
        if isinstance(doc, dict) and "spis" in doc:
            return _subset(list(doc["spis"]))
        raise ValueError(f"Unrecognised SPI config format: {path}")

    raise TypeError(f"spis must be str, list, or None; got {type(spis)}")


def _write_pruned_config(specs: dict[str, Spec]) -> str:
    """
    Write a temporary pyspi config containing exactly ``specs`` and return its
    path.  Only the modules actually needed are referenced, so broken modules
    are never imported.
    """
    config: dict[str, dict] = {}
    for module_name, fcn, params in specs.values():
        mod = config.setdefault(module_name, {})
        entry = mod.setdefault(fcn, {"configs": []})
        entry["configs"].append(dict(params))

    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="brain_spi_cfg_")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(subject_data: NDArray, spi_name: str) -> str:
    """Cache key per (subject data, SPI identifier).

    The identifier (e.g. ``cov_GraphicalLassoCV``) already encodes the SPI's
    parameters, so it is a stable config component — unlike the temp config
    file path, which changes every run.
    """
    config_hash = hash_string(spi_name)
    data_hash = hash_array(subject_data)
    return hashlib.sha256((config_hash + data_hash).encode()).hexdigest()


def _cache_path(cache_dir: Path, subject_idx: int, spi_name: str, key: str) -> Path:
    fname = f"sub{subject_idx:04d}_{spi_name}_{key[:16]}.npz"
    return cache_dir / fname


def _try_load_cache(path: Path) -> NDArray | None:
    if path.exists():
        try:
            return np.load(path)["matrix"]
        except Exception:
            return None
    return None


def _save_cache(path: Path, matrix: NDArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, matrix=matrix)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_spi_matrices(spi_name: str, matrices: NDArray, nan_frac_warn: float = 0.2) -> None:
    """Emit warnings for degenerate SPI outputs."""
    nan_frac = np.isnan(matrices).mean()
    if nan_frac > nan_frac_warn:
        warnings.warn(
            f"SPI '{spi_name}': {nan_frac:.1%} NaN values — consider excluding.",
            stacklevel=3,
        )
        return

    off_diag = matrices[:, ~np.eye(matrices.shape[1], dtype=bool)]
    if off_diag.size > 0 and np.nanstd(off_diag) < 1e-10:
        warnings.warn(
            f"SPI '{spi_name}': off-diagonal values are effectively constant — "
            "results will be uninformative.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_subject(
    subject_data: NDArray,
    configfile: str,
) -> dict[str, NDArray]:
    """
    Run pyspi Calculator on one subject using a pruned config.

    Parameters
    ----------
    subject_data : (C, T) — pyspi convention
    configfile   : path to a pruned pyspi yaml (contains exactly the wanted SPIs)

    Returns
    -------
    dict mapping spi identifier -> (C, C) ndarray
    """
    if Calculator is None:
        raise ImportError("pyspi is required for SPI computation. Install it first.")

    # Suppress pyspi's own stdout chatter *and* its per-call tqdm bar (written
    # to stderr) so we can show a single progress bar per SPI instead.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        calc = Calculator(dataset=subject_data, configfile=configfile)
        calc.compute()

    result: dict[str, NDArray] = {}
    for name in calc.spis:
        try:
            result[name] = calc.table[name].to_numpy().astype(float)
        except KeyError:
            warnings.warn(f"SPI '{name}' not found in Calculator output — skipping.")
    return result


def compute_all(
    data: NDArray,
    spis: str | list[str] | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
    progress: bool = True,
) -> dict[str, NDArray]:
    """
    Compute SPI matrices for all subjects.

    Parameters
    ----------
    data      : (B, T, C)
    spis      : SPI selection (see BrainSPI / resolve_specs).
    cache_dir : directory for per-(subject x SPI) .npz cache files.
    progress  : show one tqdm bar per SPI, advancing over all subjects.

    Returns
    -------
    dict mapping spi identifier -> (B, C, C) ndarray, in the requested order.

    Notes
    -----
    SPIs are computed one at a time across *all* subjects before moving to the
    next SPI, and the wall-time per SPI is logged at INFO level. This makes it
    easy to spot which SPIs are slow so they can be dropped from the selection.
    """
    import time

    data = np.asarray(data, dtype=float)
    B, T, C = data.shape

    if cache_dir is None:
        from .config import CACHE_DIR_DEFAULT
        cache_dir = Path(os.path.expanduser(CACHE_DIR_DEFAULT))
    cache_dir = Path(cache_dir)

    specs = resolve_specs(spis, cache_dir)
    spi_names = list(specs.keys())
    if not spi_names:
        raise ValueError("No valid SPIs to compute after resolving selection.")

    result: dict[str, NDArray] = {}
    tqdm = _get_tqdm() if progress else None
    n_spis = len(spi_names)

    for s_idx, name in enumerate(spi_names):
        # One pruned config per SPI; reused across all subjects (config does
        # not depend on subject data).
        cfg_path: str | None = None
        matrices: list[NDArray] = []
        n_computed = 0
        t0 = time.perf_counter()

        subject_iter = range(B)
        bar = None
        if tqdm is not None:
            bar = tqdm(subject_iter, total=B, desc=f"[{s_idx + 1}/{n_spis}] {name}",
                       unit="subj", leave=True)
            subject_iter = bar

        try:
            for i in subject_iter:
                subject_data = data[i].T  # (C, T)
                key = _cache_key(subject_data, name)
                cpath = _cache_path(cache_dir, i, name, key)

                mat = _try_load_cache(cpath) if use_cache else None
                if mat is None:
                    if cfg_path is None:
                        cfg_path = _write_pruned_config({name: specs[name]})
                    computed = compute_subject(subject_data, configfile=cfg_path)
                    mat = computed.get(name)
                    if mat is None:
                        mat = np.full((C, C), np.nan)
                    elif write_cache:
                        _save_cache(cpath, mat)
                    n_computed += 1

                matrices.append(mat)
        finally:
            if bar is not None:
                bar.close()
            if cfg_path is not None:
                try:
                    os.remove(cfg_path)
                except OSError:
                    pass

        elapsed = time.perf_counter() - t0
        logger.info(
            "[%d/%d] SPI '%s': %d/%d computed, %d from cache — %.1fs",
            s_idx + 1, len(spi_names), name, n_computed, B, B - n_computed, elapsed,
        )

        stack = np.stack(matrices, axis=0)  # (B, C, C)
        validate_spi_matrices(name, stack)
        result[name] = stack

    return result
