# brain-spi — Library Design Plan

Status: draft · 2026-06-09

## Goal

Wrap the SPI connectivity-analysis pipeline currently scattered across `01_pcc.ipynb`
into a small, importable Python library with a coherent API. The library should let a
user run the full pipeline in one call and then inspect either the aggregated
cross-SPI result or any individual SPI's intermediate artifacts.

Target use:

```python
from brain_spi import BrainSPI

pipe   = BrainSPI()                          # sensible defaults
result = pipe.fit(data, labels)              # data: (B, T, C); labels: (B,)

# headline result — aggregate is computed lazily on first access and cached
result.aggregate.mean_and                    # (C, C) float, cross-SPI mean of AND masks
result.aggregate.plot()                      # one-shot poster figure

# inspection
result['kendalltau'].mean_matrix()           # mean connectivity (across subjects)
result['kendalltau'].and_mask                # AND of significant-p and RF-important edges
result['kendalltau'].plot_triptych()         # 3-panel per-SPI figure

# robustness
result.bootstrap(n=20, frac=0.66)            # subject resampling — reproducibility
result.label_shuffle(n=100)                  # true null distribution
```

## Scope (what's in / out)

**In.** Connectivity derivation (via pyspi + a small custom set), per-SPI group analysis
(t-test + RF feature importance + AND intersection), cross-SPI aggregation, bootstrap
and label-shuffle nulls, plotting helpers, on-disk caching of pyspi computations.

**Out (for now).** Multi-class extensions (only 2-class HC vs. patient supported),
non-parametric group tests beyond t-test, network-level statistics (NBS etc.),
GPU acceleration. The architecture should accommodate them later, but no
provisional hooks or abstractions are added now.

---

## Design overview

### Input contract

- `data`: ndarray, shape `(B, T, C)` — `B` subjects, `T` timepoints, `C` channels/ROIs.
- `labels`: ndarray, shape `(B,)` — must contain exactly two unique values; the lower-sorted
  one is treated as group 0 ("controls"), the higher as group 1 ("patients"). Custom
  group names can be passed via `group_names=('HC', 'SZ')`.

This is the conventional Pytorch/sklearn batch-first layout. It matches `corrcoef_batch`
in `src/utils.py`. Internally the pyspi Calculator wants `(C, T)`, so we transpose per subject.

### Pipeline stages

```
                                   ┌─ t-test ─→ p_thresh ─┐
data, labels  ──→  SPI calculator ─┤                       ├─→ AND mask  ─→  aggregate
                                   └─ RF      ─→ rf_mask  ─┘                     │
                                                                                 ▼
                                                                            mean_and
                                                                            (and optionally
                                                                             null distributions)
```

1. **SPI computation.** For each subject, run pyspi `Calculator` with a curated config
   producing one (C, C) matrix per SPI. Store as a per-SPI array of shape `(B, C, C)`.
2. **Per-SPI group analysis.** For each SPI:
   - Compute group means.
   - Run Welch's t-test on each unique edge (`tril_indices`, k=-1), get t-statistic and p-value.
   - Apply Bonferroni correction at α = 0.05 / n_edges to produce `p_thresh` (bool mask).
   - Train a Random Forest classifier (500 trees) on flattened upper-triangle features; produce
     `rf_importance` (float) and `rf_mask` (bool) by thresholding at the same density
     as `p_thresh` ("matched" mode — the only one supported in v0).
   - Compute `and_mask = p_thresh & rf_mask`.
3. **Aggregation.** Across all SPIs:
   - `mean_and = mean over SPIs of and_mask.astype(float)`  — value in [0, 1] = fraction of
     SPIs that flagged that edge.
   - Also store mean of `p_thresh` and `rf_mask` separately for completeness.
4. **Null distributions (on demand).**
   - **Bootstrap subsets.** Sample `n` random subsets of `frac` of subjects; rerun stages 2–3
     on each. Returns distribution of `mean_and` matrices and per-edge survival rates.
     This is what `01_pcc.ipynb` cell 30 actually does, despite being named "permutation".
   - **Label-shuffle null.** Shuffle labels `n` times; rerun stages 2–3. Returns a true
     null distribution to compute permutation p-values against the observed `mean_and`.

### Output object

`PipelineResult` exposes:

| Attribute / method                                 | Description                                                                                |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `result['spi_name']`                             | A `PerSPIResult` (see below)                                                             |
| `result.spis`                                    | List of SPI names                                                                          |
| `result.aggregate`                               | An `AggregateResult` with `mean_and`, `mean_p_thresh`, `mean_rf_mask`, `.plot()` |
| `result.bootstrap(...)`                          | Returns a `NullDistribution` object                                                      |
| `result.label_shuffle(...)`                      | Same                                                                                       |
| `result.to_pickle(path)` / `load_pickle(path)` | Serialize / deserialize                                                                    |

`PerSPIResult` exposes:

| Attribute / method                     | Description                                              |
| -------------------------------------- | -------------------------------------------------------- |
| `matrices`                           | `(B, C, C)` array of per-subject connectivity matrices |
| `mean_matrix()`                      | `(C, C)` mean across subjects                          |
| `mean_matrix(group=0)`               | `(C, C)` mean within one group                         |
| `t_stat`, `p_value`, `p_thresh`  | Per-edge t-test artifacts                                |
| `rf_importance`, `rf_mask`         | Per-edge RF artifacts                                    |
| `and_mask`                           | Intersection                                             |
| `plot_triptych(ax=None, save=None)`  | Sig-p / RF / AND 3-panel figure                          |
| `plot_mean(ax=None, by_group=False)` | Mean connectivity figure                                 |

`AggregateResult` exposes (all properties are lazy + cached on first access):

| Attribute / method                   | Description                         |
| ------------------------------------ | ----------------------------------- |
| `mean_and`                           | `(C, C)` float, cross-SPI mean of AND masks |
| `mean_p_thresh`, `mean_rf_mask`      | `(C, C)` floats |
| `plot(ax=None, vmax='auto'\|float)`  | Single-panel aggregate figure |

`result.aggregate` itself is a cached property on `PipelineResult` — no work happens
until you ask for it. `fit()` returns as soon as per-SPI work finishes.

### SPI configuration

Default SPI list ships as a YAML file inside the package
(`brain_spi/configs/spis_default.yaml`), with `config.DEFAULT_SPIS` reading it once at
import time. The default tier contains exactly the 9 SPIs used in the notebook gallery
and the OHBM poster — the post-hoc "non-garbage" set, one representative per family,
with cov / prec using **one solver each** rather than the within-family averaging the
notebook did:

```yaml
# brain_spi/configs/spis_default.yaml
spis:
  - cov_GraphicalLassoCV       # covariance       — CV-regularised, robust default
  - prec_GraphicalLassoCV      # precision        — same family, inverse
  - spearmanr                  # rank
  - kendalltau                 # rank
  - xcorr_mean_sig-True        # spectral / cross-correlation
  - dcorr                      # distance dependence
  - mgc                        # multiscale graph correlation
  - hsic                       # kernel dependence
  - pec                        # power envelope correlation
```

Rationale for `GraphicalLassoCV` over `EmpiricalCovariance`: it picks regularisation
strength via cross-validation, so it doesn't need tuning and gives stable estimates on
short fMRI series.

A second tier — `brain_spi/configs/spis_all.yaml` — is a copy of the existing
`custom_config.yaml`, exposing the full broader pyspi set for users who want to
explore beyond the curated default. Selection is by path or by name:

```python
pipe = BrainSPI()                              # uses spis_default.yaml
pipe = BrainSPI(spis='spis_all')               # selects packaged YAML by stem
pipe = BrainSPI(spis='/path/to/my_spis.yaml')  # arbitrary path
pipe = BrainSPI(spis=['cov_EmpiricalCovariance', 'cov_LedoitWolf'])  # inline list
```

**SPI validation pass.** The 9-SPI default is curated against FBIRN. On a new dataset
some SPIs might return NaN-heavy or constant matrices. After `spis.compute_subject`,
the library runs a validation check on each SPI's output and emits a warning (not an
error) listing any SPIs whose matrices look degenerate. Users can then swap them out
of the active list without having to re-derive the curated set from scratch.

### Caching

pyspi Calculator runs are several seconds per subject per SPI, so they dominate wall time.
Cache strategy:

- Default cache dir: `~/.cache/brain_spi/` (override via `BrainSPI(cache_dir=...)`).
- Cache key: SHA-256 of `(spi_config_hash, data_array_bytes)` per subject. Reusing the same
  data with a different SPI subset costs only the new SPIs.
- One file per (subject × SPI) avoids reloading huge calculator objects when the user only
  needs a subset.
- `pipe.fit(data, labels, use_cache=True, write_cache=True)` defaults; `use_cache=False`
  forces recomputation.

---

## Package structure

```
brain_spi/
├── __init__.py          # re-exports: BrainSPI, PipelineResult, plot_* helpers
├── pipeline.py          # BrainSPI class, fit(), result assembly
├── spis.py              # pyspi config building, SPI computation per subject, cache I/O,
│                        # post-compute validation pass
├── stats.py             # ttest_edges() (Welch + Bonferroni), rf_features() (matched threshold)
├── aggregate.py         # AggregateResult, bootstrap, label-shuffle
├── viz.py               # plot_heatmap (with domain guides), triptych, aggregate panel,
│                        # multi-row grid (the poster figure)
├── domains.py           # Neuromark / generic ROI-domain definitions, guide computations
├── config.py            # DEFAULT_SPIS (loaded from YAML), DEFAULT_RF_KW, DEFAULT_T_KW
├── configs/
│   ├── spis_default.yaml    # the 9 curated SPIs (poster set)
│   └── spis_all.yaml        # broader pyspi set (existing custom_config.yaml)
└── _utils.py            # symmetrise, tril/triu helpers, hashing for cache keys

tests/
├── test_pipeline_smoke.py   # tiny synthetic dataset end-to-end
├── test_stats.py            # t-test correctness, RF determinism with seed
├── test_aggregate.py        # bootstrap shape/coverage, label-shuffle null calibration
└── test_viz.py              # smoke tests that figures render without error

examples/
├── fbirn_quickstart.ipynb   # reproduces 01_pcc.ipynb end-to-end with the new API
└── custom_spi_panel.ipynb   # using a user-defined SPI list

docs/
├── api.md
└── methodology.md           # what each pipeline stage does, references
```

### Notebook → library mapping

| Notebook cell                                           | Goes to                                                       |
| ------------------------------------------------------- | ------------------------------------------------------------- |
| `corrcoef_batch` (src/utils.py)                       | `_utils.corrcoef_batch` (kept for the PCC baseline)         |
| Cell 2 (domain setup)                                   | `domains.from_csv`                                          |
| Cell 7 `ttest`, `analyze_group_differences`         | `stats.ttest_edges`                                         |
| Cell 7 `plot_heatmap`, `plot_stats`, `plot_means` | `viz.plot_heatmap`, etc.                                    |
| Cell 11 `forest_features`                             | `stats.rf_features`                                         |
| Cell 14 `plot_and_matrix`                             | `viz.plot_triptych`, `viz.plot_aggregate`                 |
| Cell 16–17 pyspi Calculator loop                       | `spis.compute_subject`, `spis.compute_all` (with caching) |
| Cell 23 `compute_stats`                               | `pipeline.BrainSPI.fit` (split into helpers)                |
| Cell 30 `permute_data`                                | `aggregate.bootstrap` *(renamed for honesty)*             |
| Cell 32 cross-SPI averaging                             | `aggregate.mean_and`                                        |
| Cell 35 family averaging                                | **removed** — single-solver design replaces it         |

---

## Implementation tasks

A reasonable ordering. Each task should be small enough for one PR.

- **T1 — package skeleton.** `pyproject.toml` (PEP 621), package layout under `brain_spi/`,
  CI scaffold (`pytest`, `ruff`, `mypy --strict` on `stats` and `_utils`). Ship
  `configs/spis_default.yaml` and `configs/spis_all.yaml` as package data.
- **T2 — stats core.** `stats.ttest_edges` (Welch + Bonferroni, no FDR option),
  `stats.rf_features` (500 trees, matched-density threshold), `_utils` helpers. Pure
  functions, fully tested on synthetic data (no fMRI dependency yet).
- **T3 — viz core.** `viz.plot_heatmap` with domain guides, `viz.plot_triptych`,
  `viz.plot_aggregate`, optional Inter-font setup. Snapshot tests via `pytest-mpl`.
- **T4 — SPI computation + cache + validation.** `spis.compute_subject`,
  `spis.compute_all`, the post-compute validation pass that warns on degenerate
  matrices, and the cache layer keyed on data/SPI hash. Small smoke test using
  3 subjects × 3 SPIs.
- **T5 — pipeline assembly.** `BrainSPI`, `PipelineResult` (with lazy `aggregate`
  property), `PerSPIResult`, `AggregateResult`. End-to-end smoke test with synthetic
  `(B, T, C)` data. **API surface is frozen after this task** — subsequent tasks
  must not change the public signatures.
- **T6 — null distributions.** `aggregate.bootstrap`, `aggregate.label_shuffle`.
  Calibration tests: bootstrap distribution should be roughly Gaussian around the
  observed `mean_and`; label-shuffle null mean should be near zero.
- **T7 — example notebook.** Port `01_pcc.ipynb` to use the new API end-to-end. Verify
  numerics match within tolerance (Bonferroni-significant edge set should be identical;
  RF mask should match given `random_state`).
- **T8 — docs.** `docs/methodology.md` with citations, `docs/api.md` autogenerated.
  Update `README.md` with install + quickstart. No CLI; notebook-only workflow.

Bigger questions to settle along the way:

- **Disk format for cached pyspi calculators.** Currently `dill` (per-subject pickle).
  Heavy and version-fragile. Consider switching to per-SPI `.npz` (just the matrix array)
  once SPI keys are stable — much smaller and portable.

---

## Migration notes / behaviour changes from notebook

- **No more solver-averaging for cov/prec.** Single solver each (default `GraphicalLassoCV`).
  This changes the headline figure quantitatively — expect different sparsity numbers, though
  the visual story should be similar.
- **"Permutation" renamed.** Cell 30's subject-subset routine becomes
  `result.bootstrap(...)`. The poster's "Permutation null" panel was actually a 2/3 subject
  resample — the new API distinguishes this from a real `label_shuffle()` null.
- **Caching is on by default.** First `fit` is slow (~minutes); subsequent fits with the same
  data are near-instant.
- **Plotting decoupled from compute.** The notebook intermixes `plt.show()` with stats
  computation. In the library, `fit()` does no plotting; all plots are methods on the result.

---

## Decisions (resolved)

- **API stability.** Plain dataclasses + functions, no abstract base classes or plugin
  patterns. Public surface (`BrainSPI`, `PipelineResult`, `PerSPIResult`,
  `AggregateResult`) frozen at the end of T5; no breaking changes after that.
- **Random Forest size.** `DEFAULT_RF_KW = {'n_estimators': 500, 'random_state': 0}`.
- **Interface.** Library + notebook only. No CLI in v0.
- **SPI defaults.** YAML in the package. Default tier is the 9 SPIs from the notebook
  gallery (vetted against FBIRN); a broader `spis_all.yaml` ships alongside for users
  who want to explore further. Selection by stem or arbitrary path.
- **Bonferroni only, matched RF threshold only, Welch's t-test only, binary labels
  only.** Other modes (FDR, alternative RF thresholds, multiclass) are out of v0 scope.
- **Lazy aggregate.** `PipelineResult.aggregate` is a cached property; nothing is
  computed during `fit()` beyond per-SPI artifacts.
