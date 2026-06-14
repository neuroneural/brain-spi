"""brain_spi — SPI connectivity-analysis pipeline for fMRI group comparisons."""

from .pipeline import BrainSPI, PipelineResult, PerSPIResult
from .aggregate import AggregateResult, NullDistribution
from .viz import plot_heatmap, plot_triptych, plot_aggregate, plot_mean, plot_spi_grid
from . import domains


def load_npz(path):
    """Load a :class:`PipelineResult` from a portable ``.npz`` archive.

    See :meth:`PipelineResult.to_npz`. The archive is also readable with a plain
    ``numpy.load(path)`` by users who don't have brain_spi installed.
    """
    return PipelineResult.from_npz(path)


def load_pickle(path):
    """Load a :class:`PipelineResult` from a pickle file."""
    return PipelineResult.load_pickle(path)


__all__ = [
    "BrainSPI",
    "PipelineResult",
    "PerSPIResult",
    "AggregateResult",
    "NullDistribution",
    "load_npz",
    "load_pickle",
    "plot_heatmap",
    "plot_triptych",
    "plot_aggregate",
    "plot_mean",
    "plot_spi_grid",
    "domains",
]
