# brain-spi

Comparison of pairwise statistics for fMRI connectivity estimation, and a
pipeline for finding group differences across them.

`brain_spi` wraps the SPI connectivity-analysis workflow into a small, importable
library. It lets you run the full pipeline in one call and then inspect either the
aggregated cross-SPI result or any individual SPI's intermediate artifacts.

## What it does

- **Connectivity** — derive functional network connectivity (FNC) matrices from
  multivariate fMRI time series using many pairwise statistics (SPIs), via
  [pyspi](https://github.com/DynamicsAndNeuralSystems/pyspi) plus a small custom set.
- **Group differences** — per SPI, find edges that differ between two groups using
  a Welch t-test (Bonferroni-corrected) **and** a Random-Forest importance mask, then
  intersect them.
- **Aggregate** — average the per-SPI masks into a consensus matrix: the fraction of
  SPIs that flag each edge.
- **Robustness** — subject-resampling (`bootstrap`) and label-shuffle permutation nulls.
- **Plots & caching** — heatmaps with network guides, per-SPI triptychs, the aggregate
  panel; on-disk caching of the (slow) pyspi computations.

## Install

```bash
pip install -e .            # from the repo root
# core deps: numpy, scipy, scikit-learn, matplotlib, pandas, pyyaml, pyspi
```

## Quickstart

```python
import numpy as np
from brain_spi import BrainSPI

# data: (B subjects, T timepoints, C channels/ROIs);  labels: (B,) with 2 unique values
pipe   = BrainSPI(group_names=('HC', 'patient'))   # sensible defaults (9 curated SPIs)
result = pipe.fit(data, labels)

# headline result — cross-SPI aggregate (computed lazily, cached on first access)
result.aggregate.mean_and        # (C, C) float: fraction of SPIs flagging each edge
result.aggregate.plot()          # one-shot figure

# inspect a single SPI
result['kendalltau'].mean_matrix()       # (C, C) mean connectivity across subjects
result['kendalltau'].and_mask            # significant-p AND RF-important edges
result['kendalltau'].plot_triptych()     # 3-panel: sig-p / RF / AND

# not sure what's available? every result object has a repr + .help()
result.help()
```

### Choosing SPIs

```python
BrainSPI()                                   # default 9-SPI curated set
BrainSPI(spis='spis_all')                    # the broad pyspi set
BrainSPI(spis='/path/to/my_spis.yaml')       # a config file
BrainSPI(spis=['kendalltau', 'spearmanr'])   # an explicit list
```

SPIs are computed one at a time across all subjects, with a progress bar and a
per-SPI wall-time log (enable with `logging.basicConfig(level=logging.INFO)`), so
it's easy to spot and drop slow ones.

### Robustness

```python
boot = result.bootstrap(n=20, frac=0.66)     # 20 subject-resampled cross-SPI AND maps
boot.mean                                     # average of the 20 (resampling-smoothed aggregate)
boot.survival_rate()                          # how often each edge is flagged (reproducibility)

null = result.label_shuffle(n=100)            # permutation null
null.p_value(result.aggregate.mean_and)       # per-edge permutation p-values (low = significant)
```

### Saving results

The first `fit` is slow (pyspi dominates); subsequent fits on the same data are
near-instant thanks to the on-disk cache (`~/.cache/brain_spi/`, override with
`BrainSPI(cache_dir=...)`). To save a finished result:

```python
result.to_npz('result.npz')              # portable — open with plain numpy.load, no package needed
result.to_pickle('result.pkl')           # exact object

import brain_spi
result = brain_spi.load_npz('result.npz')   # or load_pickle(...)
```

The `.npz` is a flat collection of arrays with a self-describing `README` key inside,
so collaborators without `brain_spi` can still `np.load(path)` and read everything.

## Example notebook

[`examples/abide_cobre_quickstart.ipynb`](examples/abide_cobre_quickstart.ipynb) runs
end-to-end on public data — ABIDE (controls vs. autism) by default, or COBRE
(schizophrenia) via a one-line switch. Part 1 computes SPIs and inspects their mean
connectivity; Part 2 runs the significant-differences pipeline. The datasets download
automatically via [`examples/datasets.py`](examples/datasets.py), and it runs on Colab
as-is.

The original exploratory notebooks (and the FBIRN walkthrough, which needs non-public
data) live on the [`legacy`](../../tree/legacy) branch.

See [`PLAN.md`](PLAN.md) for the full design and methodology.
