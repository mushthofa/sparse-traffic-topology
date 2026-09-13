# Data dictionary (initial)

## Bandung Google samples
Expected key columns include:
- `edge_id`, `u`, `v`, `key`
- `timestamp_label`, `timestamp_iso`, `day_name`, `period`
- `osm_length_m`, `osm_time_s`
- `google_distance_m`, `google_time_s`
- `multiplier`, `status`, `source`

Derived analysis fields should never overwrite raw columns. Preferred additions include:
- `route_ratio = google_distance_m / osm_length_m`
- `route_confidence`
- `log_stress = log(google_time_s / osm_time_s)`
- explicit QC flags and exclusion reasons

## E00 audit output

`data/processed/bandung_observed_v1.csv` retains every source observation and adds:

- `route_ratio` and `route_confidence` (missing when either route length is non-positive)
- `log_stress` (missing when either travel time is non-positive)
- `qc_included`, a boolean inclusion decision
- `qc_exclusion_reasons`, a semicolon-delimited list that preserves overlapping reasons
- `value_provenance = observed_google`, distinguishing measurements from future imputed fields

The E00 soft route-consistency bounds come from `config/default.yaml`. Self-loops, invalid
lengths/times, non-OK statuses, duplicate directed-edge/time keys, and out-of-bound route
ratios are excluded explicitly; no row is dropped from the audit output.
