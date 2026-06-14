"""Plotting helpers: heatmap with domain guides, triptych, aggregate panel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .aggregate import AggregateResult
    from .domains import DomainSpec
    from .pipeline import PerSPIResult


# ---------------------------------------------------------------------------
# Grid-line styling
# ---------------------------------------------------------------------------

def _guide_line_kwargs(guides_color: str, lw: float) -> dict:
    """
    Build axhline/axvline kwargs for domain guide lines.

    With ``guides_color='auto'`` (default) the line is drawn with a contrasting
    outline (a light core haloed in dark) so it stays visible over *any*
    background — necessary for colormaps like ``inferno`` that span near-black
    to near-white, where no single flat colour reads everywhere.

    Any explicit colour ('white', 'black', '#444', …) is drawn solid, but still
    gets a thin contrasting halo so it pops on varied backgrounds.
    """
    if guides_color == "auto":
        core, halo = "white", "black"
    else:
        core = guides_color
        # crude light/dark test to pick a contrasting halo
        try:
            r, g, b, _ = matplotlib.colors.to_rgba(core)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            halo = "black" if luminance > 0.5 else "white"
        except ValueError:
            halo = "black"
    return dict(
        color=core,
        lw=lw,
        # Keep the halo thin: a wide line eats a fixed amount off every band, so
        # narrow bands (e.g. a 2-ROI network) lose a large *fraction* and look
        # squished. A light halo gives contrast without consuming cell area.
        path_effects=[patheffects.withStroke(linewidth=lw + 0.7, foreground=halo)],
    )


# ---------------------------------------------------------------------------
# Low-level heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    matrix: NDArray,
    ax: Axes | None = None,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    guides: "DomainSpec | None" = None,
    guides_color: str = "auto",
    guides_lw: float = 0.5,
    title: str | None = None,
) -> matplotlib.image.AxesImage:
    """
    Draw a square connectivity matrix as a colour-mapped image.

    Parameters
    ----------
    guides_color : 'auto' (default) draws light lines with a dark halo so they
        read on any colormap; or pass an explicit matplotlib colour.

    Returns the AxesImage (for colorbar attachment).
    """
    if ax is None:
        _, ax = plt.subplots()

    n = matrix.shape[0]
    if vmin is None and vmax is None:
        abs_max = np.nanmax(np.abs(matrix))
        vmin, vmax = -abs_max, abs_max

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
                   interpolation="nearest")

    if guides is not None:
        line_kw = _guide_line_kwargs(guides_color, guides_lw)
        for pos in guides.boundary_positions():
            ax.axhline(pos, **line_kw)
            ax.axvline(pos, **line_kw)
        ticks = guides.tick_positions()
        ax.set_xticks(ticks)
        ax.set_xticklabels(guides.names, rotation=90, fontsize=7)
        ax.set_yticks(ticks)
        ax.set_yticklabels(guides.names, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=9)

    return im


# ---------------------------------------------------------------------------
# Per-SPI triptych
# ---------------------------------------------------------------------------

def plot_triptych(
    spi_result: "PerSPIResult",
    ax: list[Axes] | None = None,
    domain_spec: "DomainSpec | None" = None,
    save: str | None = None,
    spi_name: str | None = None,
) -> Figure:
    """3-panel figure: significant-p mask / RF mask / AND mask."""
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    else:
        fig = ax[0].get_figure()

    prefix = f"{spi_name}: " if spi_name else ""

    p_float = spi_result.p_thresh.astype(float)
    rf_float = spi_result.rf_mask.astype(float)
    and_float = spi_result.and_mask.astype(float)

    for mat, title, a in [
        (p_float, f"{prefix}Sig-p (Bonferroni)", ax[0]),
        (rf_float, f"{prefix}RF mask", ax[1]),
        (and_float, f"{prefix}AND mask", ax[2]),
    ]:
        im = plot_heatmap(mat, ax=a, cmap="inferno", vmin=0, vmax=1,
                          guides=domain_spec, title=title)
        fig.colorbar(im, ax=a, fraction=0.045)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Aggregate panel
# ---------------------------------------------------------------------------

def plot_aggregate(
    agg: "AggregateResult",
    ax: Axes | None = None,
    vmax: float | str = "auto",
    domain_spec: "DomainSpec | None" = None,
    save: str | None = None,
) -> Figure:
    """Single-panel figure of cross-SPI mean_and."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    mat = agg.mean_and
    _vmax = float(np.nanmax(mat)) if vmax == "auto" else float(vmax)

    im = plot_heatmap(mat, ax=ax, cmap="inferno", vmin=0, vmax=_vmax,
                      guides=domain_spec, title="Cross-SPI mean AND")
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Mean connectivity panels
# ---------------------------------------------------------------------------

def plot_mean(
    spi_result: "PerSPIResult",
    ax: Axes | None = None,
    by_group: bool = False,
    group_names: tuple[str, str] = ("Group 0", "Group 1"),
    domain_spec: "DomainSpec | None" = None,
    spi_name: str | None = None,
) -> Figure:
    """Mean connectivity figure for one SPI, optionally split by group."""
    if by_group:
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        else:
            fig = ax[0].get_figure()
            axes = ax
        for gi, (gname, a) in enumerate(zip(group_names, axes)):
            mat = spi_result.mean_matrix(group=gi)
            prefix = f"{spi_name}: " if spi_name else ""
            im = plot_heatmap(mat, ax=a, guides=domain_spec, title=f"{prefix}{gname} mean")
            fig.colorbar(im, ax=a, fraction=0.045)
    else:
        if ax is None:
            fig, single_ax = plt.subplots(figsize=(6, 5))
        else:
            single_ax = ax
            fig = ax.get_figure()
        prefix = f"{spi_name}: " if spi_name else ""
        mat = spi_result.mean_matrix()
        im = plot_heatmap(mat, ax=single_ax, guides=domain_spec, title=f"{prefix}mean connectivity")
        fig.colorbar(im, ax=single_ax, fraction=0.045)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-row grid (poster figure)
# ---------------------------------------------------------------------------

def plot_spi_grid(
    results: dict[str, "PerSPIResult"],
    domain_spec: "DomainSpec | None" = None,
    save: str | None = None,
) -> Figure:
    """One triptych row per SPI — the poster-style overview figure."""
    spis = list(results.keys())
    n = len(spis)
    fig, axes = plt.subplots(n, 3, figsize=(14, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(spis):
        plot_triptych(results[name], ax=axes[row], domain_spec=domain_spec, spi_name=name)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig
