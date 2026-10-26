# METR-LA local sources

Use the original DCRNN measurement distribution and pinned sensor graph documented
in [the E01 benchmark contract](../../../docs/metr_la_benchmark.md).

Download `metr-la.h5` from the publisher-linked Google Drive folder, then run:

```bash
python scripts/download_metrla.py --measurements /path/to/metr-la.h5
```

Record known original download dates with `--measurement-download-date YYYY-MM-DD`.
The helper records URLs, versions, SHA256, sizes, registration dates and known
original download dates/terms in `manifest.json`. Unknown original dates and
independent measurement redistribution terms remain explicit. No API credentials
are needed for local-file ingestion. Keep source files immutable and out of Git.

The previous placeholder referenced Zenodo record 5146275; E01 uses the explicitly
traceable DCRNN distribution instead and makes no equivalence claim for that record.
