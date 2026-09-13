# Data policy

## Raw
`data/raw/` contains immutable first-party/source snapshots. Never edit in place.

## External
`data/external/` contains third-party public benchmark data such as METR-LA. Record the source URL, DOI/record ID, download date, license/terms, and SHA256 in a manifest.

## Interim / processed
These directories contain reproducible outputs and are ignored by Git except for placeholder files.

## Large files
The current Bandung `full_dynamic_edges.csv` and dynamic GraphML are stored compressed. `.gitattributes` is prepared for Git LFS. If the team prefers DVC or an institutional object store, replace the LFS rule but retain immutable manifests and hashes.
