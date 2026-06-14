"""Internal utilities: array helpers, hashing for cache keys."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray


def symmetrise(m: NDArray) -> NDArray:
    """Return (m + m.T) / 2, preserving diagonal."""
    return 0.5 * (m + m.T)


def tril_indices(n: int) -> tuple[NDArray, NDArray]:
    """Lower-triangle (excluding diagonal) indices for an (n, n) matrix."""
    return np.tril_indices(n, k=-1)


def tril_vec(m: NDArray) -> NDArray:
    """Extract lower-triangle vector from a square matrix."""
    idx = tril_indices(m.shape[-1])
    return m[..., idx[0], idx[1]]


def tril_to_matrix(v: NDArray, n: int) -> NDArray:
    """Reconstruct a symmetric matrix from a lower-triangle vector."""
    m = np.zeros((n, n), dtype=v.dtype)
    idx = tril_indices(n)
    m[idx] = v
    m = m + m.T
    return m


def corrcoef_batch(x: NDArray, eps: float = 1e-8) -> NDArray:
    """
    Batched Pearson correlation over features for time-series data.

    Parameters
    ----------
    x : array_like, shape (B, T, D)
    Returns
    -------
    C : ndarray, shape (B, D, D)
    """
    x = np.asarray(x, dtype=float)
    B, T, D = x.shape
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    Z = (x - mu) / (sd + eps)
    C = np.einsum("btd,bte->bde", Z, Z) / T
    C = 0.5 * (C + C.transpose(0, 2, 1))
    bi = np.arange(B)[:, None]
    di = np.arange(D)
    C[bi, di, di] = 1.0
    return C


def hash_array(arr: NDArray) -> str:
    """SHA-256 hex digest of a numpy array (dtype + shape + bytes)."""
    h = hashlib.sha256()
    h.update(arr.dtype.str.encode())
    h.update(np.array(arr.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def hash_string(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()
