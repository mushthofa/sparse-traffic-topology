# Experiment registry

| ID | Experiment | Primary outputs |
|---|---|---|
| E00 | Bandung data audit + route QC | QC table, exclusion report, distributions |
| E01 | METR-LA dense benchmark preparation | graph + dense signal matrix |
| E02 | Bandung baseline reconstruction | Model-A benchmark metrics |
| E03 | Graph reconstruction CV | MAE/RMSE/R2 by sample density |
| E04 | Spatial-block / road-class CV | generalization metrics |
| E05 | Topology fidelity under sparse sensing | diagram distances, Betti curves, pTHI error |
| E06 | Bandung 21-state persistent topology | D0/D1, TP0/TP1, pTHI, distance matrix |
| E07 | Topological road criticality | C_topo rankings/maps |
| E08 | Blockage validation | OD delay, efficiency loss, rank metrics |
| E09 | Incremental-value models | CV gain from adding C_topo |
| E10 | Sensitivity / ablation | robustness tables |
| E11 | Publication regeneration | final figures/tables |

Every experiment must save its configuration, seed, metrics, and software version information.
