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
