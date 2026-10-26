# E01 — METR-LA sparse-observation benchmark

Specification: [issue #4](https://github.com/mushthofa/sparse-traffic-topology/issues/4).
E01 prepares inputs, audits, partitions, and evaluation contracts. It does not fit
reconstruction models or compute persistent homology, pTHI, or road criticality.

METR-LA is a measured reference **only at its instrumented traffic sensors**. It is
not complete ground truth for all Los Angeles roads. Bandung has 100 Google-observed
OSM segments over 21 weekly states; model-imputed values elsewhere are not measured
ground truth. METR-LA sensor nodes and its distance-derived sensor affinity graph
are different objects from Bandung's directed OSM edges `(u, v, key)` and their
road-segment interaction graph. No mapping between those node sets is implied.

## Sources, version, units, and access

Use the original [DCRNN distribution](https://github.com/liyaguang/DCRNN), associated
with Li et al., *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
Forecasting*, ICLR 2018, [arXiv:1707.01926](https://arxiv.org/abs/1707.01926).

- Measurements: `metr-la.h5` from the [publisher-linked Google Drive folder](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX).
  The DCRNN README also links Baidu, but no alternate mirror is silently selected.
- Graph: `data/sensor_graph/adj_mx.pkl` and `graph_sensor_ids.txt`, pinned to DCRNN
  commit `602afd9d767d3aa1c9b3eac51710d6aeee12c227`.
- Exact graph construction: [pinned gen_adj_mx.py](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/scripts/gen_adj_mx.py).
- The paper reports speeds in **miles per hour**. Original HDF values are retained;
  no conversion is performed. A local substitute must explicitly establish its
  units and provenance rather than inheriting the name METR-LA.
- DCRNN source code is MIT-licensed. An independent license for the traffic files
  was not established here; consult the publisher and applicable data-provider
  terms before redistribution. No authentic binary data is included in this PR.

Expected measurements are approximately 34,272 timestamps × 207 sensors, sampled
every five minutes. These are checks, not manufactured dimensions: the audit saves
actual shape and `expected_shape_matches`; mismatch is visible and does not pad,
truncate, or relabel a local source. The Drive file has no publisher checksum
verified here or immutable release identifier. Record the acquired file's digest
and acquisition version, and pin that digest in a local YAML config for later runs.
The default graph hash is pinned:

| File | SHA256 |
|---|---|
| `adj_mx.pkl` | `a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3` |
| `graph_sensor_ids.txt` | `3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990` |

The acquisition helper verifies these hashes before use. It refuses to overwrite
existing differing sources and records URLs, versions, sizes, hashes, registration dates, known download
dates, and license/terms status in `manifest.json`. Unknown original dates are null;
use `--measurement-download-date YYYY-MM-DD` when registering a known local download. A locally computed measurement digest identifies bytes; it is not
an independent guarantee of publisher authenticity. `expected_sha256.measurements`
in the processing config can enforce a previously trusted digest. All configured
source hashes are recorded before deserialization, and configured expectations
are checked before loading.

## Download and run locally

From the repository root, with Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# Download metr-la.h5 manually from the publisher's Drive folder above.
# This command downloads the pinned graph and registers the local measurement file:
python scripts/download_metrla.py --measurements /absolute/path/to/metr-la.h5
python scripts/prepare_metr_la.py --config config/metr_la.yaml --trust-graph-pickle
python scripts/audit_metr_la.py --config config/metr_la.yaml --trust-graph-pickle
ruff check src tests scripts
pytest -q
```

For an already populated directory, acquisition is unnecessary. Set the paths,
source version/URL, units, and expected hashes in a local copy of
`config/metr_la.yaml`; then pass `--config /path/to/local.yaml`. Relative data paths
are resolved against the repository root, or against the explicit `--root PATH`.
`hdf_key: null` requires a single HDF dataset; set a key explicitly for multikey
files. PyTables (`tables`) is installed for pandas' DCRNN HDF reader.

Only deserialize trusted HDF5 and pickle files: pandas HDF stores may contain
pickled objects, and a graph pickle requires the explicit `--trust-graph-pickle`
flag. For local text inputs, the measurement CSV format is
`timestamp,sensor_id_1,...` and graph CSV is a square matrix with both row and column
ID labels. ID strings, including leading zeroes, are preserved. No HDF-derived
train/val/test NPZ from an unrelated forecasting split is substituted for original
measurements.

Inputs are read-only. Acquisition creates new sources under `data/external/metr-la/`;
subsequent processing never changes them or `data/raw/`. Raw HDF, pickle, NPZ and
Parquet sources are ignored by Git, as are all generated `data/processed/` arrays.
Do not commit authentic binaries even when using another local path.

## Timestamps and speed validity

The source timestamp index and sensor column order are preserved exactly. There
is no sorting, gap filling, resampling, interpolation, or timezone conversion.
Naive timestamps retain their source wall-clock interpretation; Los Angeles local
time is the documented assumption, **not a verified UTC/DST assignment**. Ambiguous
DST hours cannot be resolved without external provenance. Offset-aware inputs
retain their encoded timezone. Hour/day summaries use the source clock, including
this limitation. No UTC claims are made for naive inputs.

Timestamp audit records bounds, timezone representation, NaT, duplicate rows and
excess duplicates, source-order interval counts, off-grid observations, and missing
points on the five-minute grid from minimum to maximum source time. The audit can
describe duplicates and unsorted inputs, but preparation refuses missing,
duplicate, or nonchronological timestamps. Gaps and irregular positive intervals
are recorded and retained; E01 does not form time windows across gaps.

Let `Y[t,s]` be the unchanged source speed and `V[t,s]` its validity indicator.
By default `V = isfinite(Y) AND (Y > 0)`. NaN, positive/negative infinity, finite
negative values, and zero values are counted separately. A zero is ambiguous
because source processing can use it as a missing-value code; it is not asserted
to be a physically observed stationary vehicle. `zero_policy: include` is an
explicit sensitivity alternative **only when source provenance supports it**.
No invalid value is replaced by zero. All source rows remain in evaluator storage;
unavailable model inputs are NaN with an explicit mask.

Completeness uses the recorded source rows as its denominator; missing timestamp
slots are reported separately and are not silently synthesized. Per-sensor tables
include valid counts/completeness, mean, median, population
standard deviation (`ddof=0`), CV (`std/mean`), and validity-reason counts. Constant
means one distinct valid value with at least one valid reading; nearly constant
means at least two valid readings and CV ≤ 0.01 by default. An all-invalid sensor
has undefined statistics and no constant-data claim. CV is undefined at mean zero.
Hourly, day-of-week, and hour-of-week tables report both total and valid entries;
their sample standard deviations use `ddof=1`. Network distributions use only
valid speeds and are descriptive audits, **never model normalization statistics**.

## Graph identity and semantics

The DCRNN pickle is `[sensor_ids, sensor_id_to_index, adjacency]`. The mapping must
exactly enumerate its IDs, the matrix must be square, and every ID must be unique,
nonmissing, and present in the measurement columns. Labeled CSV row/column IDs
must have the same order. Finite nonnegative weights are required.

An order mismatch fails by default. `--allow-graph-permutation` explicitly permits
alignment when the ID sets agree: for each measurement sensor, record its source
index `p`, then use **both** axes `A_aligned = A_source[p,p]` via `np.ix_`.
The original file and source-ID ordering are retained in provenance. Missing IDs,
duplicate IDs, inconsistent pickle mappings, and row/column ID disagreements fail;
they are never repaired by guessing.

DCRNN graph entries are thresholded affinities `exp(-(distance / sigma)^2)` derived
from road-network distances; the supplied matrix is preserved. They are not raw
metric distances or travel-time measurements. No graph construction from traffic
targets is performed. Directed nonzero edges and source self-loops are retained.
For future topology work, a separate undirected graph takes the union of directions
with `max(A_ij, A_ji)` and removes the diagonal. This is an explicit affinity
symmetrization, not a physical road reconstruction or a filtration definition.
A distance-weighted source is rejected until a suitable separate policy exists.

Outputs include directed and undirected edge tables, degree/strength/component
membership tables, weak/strong/undirected component counts, isolated nodes ignoring
self-loops, densities excluding loops, and degree/weight distributions. Both edge
counts with and without source loops are named explicitly.

## Chronological and spatial protocols

Time partitions are `[0, floor(.7*T))` train, the next `floor(.1*T)` rows validation,
and all remaining rows test. No shuffle. Each partition must be nonempty. Counts,
exclusive stop indices, inclusive start indices, and first/last timestamps are saved.
There is no overlapping window construction in E01. E02 must keep complete target
windows inside the assigned partition and respect causal input boundaries.

For seeds 42, 43, 44, 45, 46 independently rank sensor IDs by SHA256 of the compact
UTF-8 JSON array `[seed, sensor_id]` (ties broken by exact ID). This is a deterministic
seeded pseudorandom ranking, independent of source axis order and NumPy RNG version.
Take the first `floor(f*N)` IDs for each `f` in .2, .4, .6, .8. Counts must leave
both observed and withheld sensors. Small graphs may have coincident subsets across
independent seeds; independence does not imply guaranteed disjoint or unique sets.
For N=207, nominal observed counts would be 41, 82, 124, 165; these are not claimed
as verified measurement dimensions here. Selected IDs/counts and **actual** fractions
are saved. Within each seed the four selections are nested, and every subset is
fixed across time. No mask is selected based on speed values or quality metrics.

Let `S_r,f[s]` indicate a selected sensor and `I_q[t]` indicate a temporal partition.
The model observation mask `O` specifies which target values may enter inputs:

- **A — inductive spatial holdout (primary):** `O[t,s] = V[t,s] AND S[s]` for all
  timestamps. Withheld sensor targets are forbidden in train, validation, and test.
  The supplied static graph, sensor identities, and declared static metadata are
  available. No traffic-derived graph or feature from withheld targets is allowed.
- **B — transductive spatial masking (secondary):** `O[t,s] = V[t,s]` for training
  timestamps, and `V[t,s] AND S[s]` during validation/test. Thus withheld sensor
  histories end exactly at the exclusive training stop. There is no progressive
  reveal of their validation history. Training is the chronological block only;
  test observed covariates can be used to reconstruct the simultaneous test state
  but cannot fit parameters or choose hyperparameters. Forecasting would require
  an additional causal input-window policy and is not implemented here.

For A, held-out evaluation masks are `E_q[t,s] = I_q[t] AND V[t,s] AND NOT S[s]`.
For B, validation/test use the same masks; `E_train` is empty because those histories
are observed. Validation scores may select hyperparameters; hidden validation
labels remain evaluator-only and never become model inputs or fitting targets.
Test labels never participate in fitting, normalization, hyperparameter selection,
or traffic-derived graph construction. `O AND E_q` is empty for every scored split.

`protocol_masks` produces O and E; `permitted_measurements(Y,O)` returns a **separate**
array with NaN outside O. The evaluator archive contains full unchanged Y and V and
must remain inaccessible to model fitting. `training_normalization` uses only
`Y[train][O[train]]`: one global population mean/std for each protocol/seed/fraction,
with a scale of 1 for a constant sample and an explicit failure for no valid sample.
No hidden per-sensor statistics, audit-wide statistics, validation, or test targets
are used. There is no model-training interface that automatically consumes the archive.

## Evaluation contract for E02

`heldout_metrics(Y, predictions, E)` intersects E with target validity and computes
`MAE = sum(abs(pred-Y))/n`, `RMSE = sqrt(sum((pred-Y)^2)/n)` and n. Missing/invalid
targets do not enter numerator or denominator. Predictions must be finite at every
eligible pair; missing predictions fail rather than silently improving coverage.
Zero eligible entries returns n=0 and null errors, not zero error.
`grouped_heldout_metrics` reports every sensor and all 24 source-clock hours,
including empty groups. Shapes and boolean mask types are checked.

Attach protocol, split, fraction, repetition, and seed to each overall result and
pass five records per protocol/split/fraction to `aggregate_repetitions`. It requires
five distinct repetitions and seeds, reports equal-weight macro means/sample SDs,
nonempty repetition count, summed coverage, and entry-weighted pooled MAE/RMSE.
Pooled RMSE is the square root of pooled squared error, not the mean of run RMSEs.
Overlapping held-out pairs across repetitions count once per repetition; pooled
coverage is a sum, not a count of unique pairs. Apply the same aggregation separately
to each sensor/hour subgroup when those summaries are needed. E01 tests only
synthetic arithmetic; it publishes **no reconstruction performance results**.

Candidate E02 baselines: permitted-training global mean; historical hour-of-week
profiles fit only to permitted training data (never hidden per-sensor histories in
A); static graph-neighbor affinity-weighted interpolation with explicit fallback
for unobserved neighborhoods; and graph Laplacian regularization. These require
separate implementations and validation. Future topology comparisons must agree
on graph symmetrization, affinity-to-filtration rules and scaling fitted using only
permitted training information. E01 intentionally does not choose or implement PH.

## Artifacts and validation status

Commands write only E01-prefixed outputs:

- `results/metrics/e01_audit.json`: provenance, configuration, hashes/software,
  measured shape, timestamp/value quality, graph summaries and alignment.
- `results/metrics/e01_benchmark.json`: boundaries and all 40 protocol/mask runs,
  selected IDs, coverage and permitted-training normalization (no model scores).
- `results/tables/e01_*.csv`: sensors, temporal summaries, graph degrees, directed
  and undirected edges, plus timestamp-row mapping after preparation.
- `results/figures/e01_{speed,affinity,temporal}.svg`: diagnostic distributions/patterns.
- `data/processed/metr-la/e01_evaluator_targets.npz`: **local-only**, full original
  targets, validity, exact sensor order, and source timestamp strings. Load numeric
  arrays with `np.load(..., allow_pickle=False)`. It is evaluator-owned data.

The official Drive measurement download returned `CONNECT tunnel failed, response
403` in this environment. Authentic measurement ingestion, dimensions, timestamp
coverage and real-data QC are **unverified**. No real-data statistics are fabricated.
The smaller pinned graph files were acquired, hash-verified, and audited independently:

```bash
python scripts/download_metrla.py --destination data/external/metr-la --audit-graph
```

This writes `results/metrics/e01_source_graph.json`. Verified graph-only counts:
207 nodes, 207 source self-loops, 1,515 directed edges excluding loops, 1,313
max-affinity undirected edges, 2 weak/undirected components, 13 strong components,
and 1 isolated node when loops are ignored. The source graph's ID list matches its
separately supplied ID file. Graph access does not verify measurement alignment
or measurement quality; those fields are explicitly false in this report.

The committed reproducible demonstration uses explicitly synthetic fixtures:

```bash
python scripts/prepare_metr_la.py --config config/metr_la_synthetic.yaml
python scripts/audit_metr_la.py --config config/metr_la_synthetic.yaml
ruff check src tests scripts
pytest -q
```

Synthetic outputs use `e01_synthetic_` prefixes and visible SYNTHETIC labels, five
invented IDs, 40 rows, one deliberate timestamp gap, and separate invalid-reading
cases. They demonstrate software behavior only. Fixtures and SVG/CSV/JSON are
lightweight review artifacts; no HDF, NPZ, Parquet or other binary is committed.
Tests include synthetic HDF ingestion, graph permutation of both axes, ID failures,
timestamp defects, mask determinism/nestedness, all 20 masks under both protocols,
forbidden-value perturbations, metrics excluding invalid targets, missing-prediction
failure, and repeated CLI execution. Existing E00 tests also run; E00 artifacts are
not regenerated. Passing local checks do not establish GitHub Actions success;
that must be checked on the PR before merge.
