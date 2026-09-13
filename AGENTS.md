# AGENTS.md — Research Repository Instructions

## Mission
Develop a reproducible applied-mathematics/network-science paper on sparse urban traffic reconstruction, persistent topology, and road criticality using METR-LA and Bandung Google Maps/OSM data.

## Non-negotiable rules
1. **Never modify files under `data/raw/`.** Treat them as immutable evidence.
2. Put reusable logic in `src/traffic_thi/`; notebooks should orchestrate and visualize only.
3. Every experiment must save machine-readable metrics to `results/metrics/` and publication tables/figures to `results/tables/` and `results/figures/`.
4. Use deterministic random seeds unless an experiment explicitly studies stochasticity.
5. Do not silently drop observations. QC exclusions must be reported with counts and reasons.
6. Preserve directed OSM edge identity `(u, v, key)` throughout the pipeline.
7. Separate observed Google values from imputed/reconstructed values with explicit provenance fields.
8. Never commit credentials or API keys.
9. Add or update tests when changing mathematical/reconstruction logic.
10. Before claiming a result in the paper, ensure it is produced by a committed script/notebook and saved in `results/`.

## Mathematical conventions
- Physical road graph: `G_R=(V,E)`.
- Road-segment interaction/line graph: `H=L(G_R)`; preserve directed form for propagation and explicitly document any symmetrization used for clique-complex PH.
- Baseline OSM edge travel time: `b_e`.
- Google-observed time: `g_{e,t}`.
- Preferred stress variable: `y_{e,t}=log(g_{e,t}/b_e)` after QC.
- Reconstruction field: `X in R^{|E| x T}`.
- Compute persistence diagrams before scalar summaries such as pTHI.
- Report `TP_0`, `TP_1` separately in addition to any weighted pTHI.
- Topological criticality should preferentially use persistence-diagram distance under edge/segment perturbation; delta-pTHI is secondary.

## Coding style
- Python 3.11+.
- Type hints for public functions.
- Small pure functions where practical.
- `ruff` for linting/formatting, `pytest` for tests.
- Use pandas/NumPy/SciPy/NetworkX for data/graph work; GUDHI for PH unless a benchmark justifies another library.
- Avoid notebook-only helper functions.

## Before finishing any task
Run:

```bash
ruff check src tests scripts
pytest -q
```

If dependencies or commands change, update `README.md` and `pyproject.toml`.
