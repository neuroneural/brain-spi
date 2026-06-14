"""Group-level statistics: Welch t-test + Bonferroni, Random Forest feature importance."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _scipy_stats
from sklearn.ensemble import RandomForestClassifier

from ._utils import tril_indices, tril_vec, tril_to_matrix


def ttest_edges(
    matrices: NDArray,
    labels: NDArray,
    alpha: float = 0.05,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Welch's t-test on each lower-triangle edge, Bonferroni-corrected.

    Parameters
    ----------
    matrices : (B, C, C)
    labels   : (B,) with exactly two unique values
    alpha    : family-wise error rate before correction

    Returns
    -------
    t_stat   : (C, C) symmetric, zeros on diagonal
    p_value  : (C, C) symmetric, ones on diagonal
    p_thresh : (C, C) bool mask, True where p < alpha / n_edges
    """
    matrices = np.asarray(matrices, dtype=float)
    labels = np.asarray(labels)
    B, C, _ = matrices.shape

    groups = np.unique(labels)
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 unique labels, got {len(groups)}")

    g0 = matrices[labels == groups[0]]
    g1 = matrices[labels == groups[1]]

    idx = tril_indices(C)
    n_edges = len(idx[0])
    bonferroni_alpha = alpha / n_edges

    x0 = g0[:, idx[0], idx[1]]  # (n0, n_edges)
    x1 = g1[:, idx[0], idx[1]]  # (n1, n_edges)

    t_vec, p_vec = _scipy_stats.ttest_ind(x0, x1, axis=0, equal_var=False)

    t_mat = tril_to_matrix(t_vec, C)
    p_mat = tril_to_matrix(p_vec, C)
    np.fill_diagonal(p_mat, 1.0)

    thresh_mat = p_mat < bonferroni_alpha

    return t_mat, p_mat, thresh_mat


def rf_features(
    matrices: NDArray,
    labels: NDArray,
    density: float | None = None,
    rf_kw: dict | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Random Forest feature importance on lower-triangle edges.

    Parameters
    ----------
    matrices : (B, C, C)
    labels   : (B,)
    density  : fraction of top edges to flag; if None, uses proportion of
               significant edges in p_thresh (pass explicitly for matched mode)
    rf_kw    : kwargs forwarded to RandomForestClassifier

    Returns
    -------
    rf_importance : (C, C) symmetric importance matrix
    rf_mask       : (C, C) bool mask of top-density edges
    """
    from .config import DEFAULT_RF_KW

    matrices = np.asarray(matrices, dtype=float)
    labels = np.asarray(labels)
    B, C, _ = matrices.shape

    kw = {**DEFAULT_RF_KW, **(rf_kw or {})}

    idx = tril_indices(C)
    X = matrices[:, idx[0], idx[1]]  # (B, n_edges)

    clf = RandomForestClassifier(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, labels)

    imp_vec = clf.feature_importances_  # (n_edges,)

    imp_mat = tril_to_matrix(imp_vec, C)

    if density is None:
        density = imp_vec.size / imp_vec.size  # all — caller should always pass density

    n_top = max(1, int(round(density * len(imp_vec))))
    threshold = np.sort(imp_vec)[-n_top]
    mask_vec = imp_vec >= threshold

    mask_mat = tril_to_matrix(mask_vec.astype(float), C).astype(bool)

    return imp_mat, mask_mat
