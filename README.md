# Sparse Traffic Topology and Road Criticality

Reproducible research repository for sparse urban traffic reconstruction, persistent topology (THI/pTHI), and road vulnerability analysis using:

1. **METR-LA** as a controlled dense benchmark.
2. **Bandung Google Maps + OSM** as the sparse real-world case study.

## Core research question

Can sparse traffic observations recover the persistent topology of a large urban road network well enough to identify road segments whose disruption produces disproportionate network-wide impact?

## Repository layout

- `data/raw/` immutable source snapshots.
- `data/external/` third-party public benchmark files (METR-LA).
- `data/interim/` reproducible intermediate artifacts; not versioned.
- `data/processed/` final model-ready datasets; not versioned.
- `src/traffic_thi/` reusable research code.
- `notebooks/` thin experiment/report notebooks; logic should live in `src/`.
- `scripts/` command-line entry points for reproducible pipelines.
- `results/` saved metrics, tables, and publication figures.
- `paper/` LaTeX manuscript.
- `config/` experiment configuration.
- `tests/` unit and regression tests.

## Bandung data snapshot

The latest uploaded Bandung snapshot is stored in `data/raw/bandung/`:

- `sampled_edges.csv` — 100 sampled directed OSM edges.
- `google_edge_samples.csv` — 2,100 successful Google observations (100 edges × 21 time states).
- `full_dynamic_edges.csv.gz` — compressed dense dynamic edge table.
- `bandung_itb_rect_with_dynamic.graphml.xml.gz` — compressed dynamic OSMnx graph snapshot.

Raw files are immutable. Any cleaning, QC filtering, reconstruction, or topology transformation must write to `data/interim/` or `data/processed/`.

## Research stages

1. Data audit and route-consistency QC.
2. METR-LA benchmark preparation.
3. Sparse spatiotemporal graph-signal reconstruction.
4. Reconstruction validation and sampling-density experiments.
5. Persistent topology: persistence diagrams, Betti curves, total persistence, pTHI.
6. Topological road criticality by perturbation.
7. Road-blockage / OD travel-time / efficiency validation.
8. Robustness, ablations, and publication figures.

## Reproducibility rules

- No analysis depends on notebook execution order.
- Every experiment has a fixed random seed and a config file/CLI arguments.
- Every final figure/table must be regenerable from code.
- Never hand-edit generated CSVs, figures, or LaTeX result tables.
- API keys live in `.env` only and are never committed.
- Large derived data are reproducible outputs, not the canonical scientific record.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -e ".[dev]"
python scripts/audit_bandung.py
pytest
```

The E00 audit command reads, but never changes, the raw Bandung snapshot. It writes the
row-preserving `data/processed/bandung_observed_v1.csv`, publication-ready summaries to
`results/tables/`, machine-readable configuration and counts to `results/metrics/`, and
route-ratio and stress distributions as SVG files in `results/figures/`. Every excluded
observation has one or more explicit reasons; its directed `(u, v, key)` identity and
observed-data provenance remain in the output.

E00 reports both moderate (`0.7–1.5`) and strict (`0.8–1.25`) route-ratio QC profiles,
using the moderate profile for `bandung_observed_v1.csv`. It also checks raw-file hashes,
schema and edge/time-grid integrity, summarizes temporal log-stress variability, and
compares the sampled road-class distribution with all directed edges in the full snapshot.

See `AGENTS.md` for Codex/agent instructions and `docs/experiment_registry.md` for the planned experiment IDs.
