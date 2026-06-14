"""Package-level configuration constants."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import yaml

_PKG = "brain_spi"
_CONFIGS = Path(__file__).parent / "configs"


def _load_spi_names(stem: str) -> list[str]:
    path = _CONFIGS / f"{stem}.yaml"
    with open(path) as f:
        doc = yaml.safe_load(f)
    return doc["spis"]


DEFAULT_SPIS: list[str] = _load_spi_names("spis_default")

DEFAULT_RF_KW: dict = {"n_estimators": 500, "random_state": 0}
DEFAULT_T_KW: dict = {}  # kwargs forwarded to scipy ttest_ind

CACHE_DIR_DEFAULT: str = "~/.cache/brain_spi"
