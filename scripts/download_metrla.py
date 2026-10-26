"""Acquire pinned DCRNN graph assets and register an already downloaded HDF file.

The publisher's Google Drive folder is interactive. Download metr-la.h5 locally
first; this script never circumvents access controls or overwrites source files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from traffic_thi.io import sha256_file

COMMIT = "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
BASE = f"https://raw.githubusercontent.com/liyaguang/DCRNN/{COMMIT}/data/sensor_graph"
MEASUREMENT_URL = "https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
GRAPH_HASHES = {
    "adj_mx.pkl": "a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3",
    "graph_sensor_ids.txt": "3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990",
}


def acquire_graph(destination: Path) -> dict:
    """Download pinned, checksum-verified graph assets, or verify existing copies."""
    import hashlib

    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, expected in GRAPH_HASHES.items():
        path = destination / name
        if path.exists():
            if sha256_file(path) != expected:
                raise ValueError(f"Existing source checksum mismatch: {path}; refusing overwrite")
        else:
            with urlopen(f"{BASE}/{name}", timeout=60) as response:
                blob = response.read()
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError(f"Download checksum mismatch: {name}")
            with path.open("xb") as stream:
                stream.write(blob)
        records[name] = {
            "url": f"{BASE}/{name}",
            "sha256": expected,
            "version": COMMIT,
            "bytes": path.stat().st_size,
        }
    return records


def main() -> None:
    """Record traceable local measurements; no automatic Drive scraping or fallback mirror."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/external/metr-la"))
    parser.add_argument("--measurements", type=Path, help="Already downloaded trusted metr-la.h5")
    parser.add_argument("--measurement-url", default=MEASUREMENT_URL)
    parser.add_argument("--measurement-version", default="DCRNN Google Drive distribution")
    parser.add_argument(
        "--measurement-sha256", help="Optional previously trusted measurement digest"
    )
    parser.add_argument(
        "--audit-graph",
        action="store_true",
        help="Audit the pinned, checksum-verified graph independently of measurements",
    )
    parser.add_argument(
        "--graph-report", type=Path, default=Path("results/metrics/e01_source_graph.json")
    )
    parser.add_argument(
        "--measurement-download-date",
        help="Original measurement download date (YYYY-MM-DD), if known",
    )
    args = parser.parse_args()
    manifest = args.destination / "manifest.json"
    existing = json.loads(manifest.read_text()) if manifest.exists() else {}
    preexisting = {name: (args.destination / name).exists() for name in GRAPH_HASHES}
    records = acquire_graph(args.destination)
    if args.measurements:
        digest = sha256_file(args.measurements)
        if args.measurement_sha256 and digest != args.measurement_sha256:
            raise ValueError("Local measurement checksum mismatch")
        target = args.destination / "metr-la.h5"
        if target.exists():
            if sha256_file(target) != digest:
                raise ValueError("Existing measurements differ; refusing overwrite")
        else:
            with args.measurements.open("rb") as source, target.open("xb") as dest:
                shutil.copyfileobj(source, dest)
        if sha256_file(target) != digest:
            raise ValueError("Copied measurements failed checksum validation")
        records[target.name] = {
            "url": args.measurement_url,
            "sha256": digest,
            "version": args.measurement_version,
            "bytes": target.stat().st_size,
        }
    today = datetime.now(UTC).date().isoformat()
    for name, record in records.items():
        previous = existing.get(name, {})
        if previous and any(previous.get(key) != value for key, value in record.items()):
            raise ValueError(
                f"Provenance differs for existing source {name}; use a separate directory"
            )
        record["registered_date_utc"] = previous.get("registered_date_utc", today)
        if name in GRAPH_HASHES:
            record["download_date_utc"] = previous.get(
                "download_date_utc", None if preexisting[name] else today
            )
            record["license_terms"] = "DCRNN repository MIT license; see upstream LICENSE"
        else:
            record["download_date_utc"] = args.measurement_download_date or previous.get(
                "download_date_utc"
            )
            record["license_terms"] = (
                "Independent traffic data redistribution terms unverified; consult publisher"
            )
    manifest.write_text(json.dumps({**existing, **records}, indent=2, sort_keys=True) + "\n")
    print(f"Graph source pinned to DCRNN {COMMIT}; provenance: {manifest}")
    if args.audit_graph:
        from traffic_thi.metr_la import graph_audit, load_sensor_graph

        ids = (args.destination / "graph_sensor_ids.txt").read_text().strip().split(",")
        # Explicit --audit-graph authorizes only this hash-verified pinned pickle.
        graph = load_sensor_graph(args.destination / "adj_mx.pkl", ids, trust_pickle=True)
        report, _, _ = graph_audit(graph)
        args.graph_report.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "experiment_id": "E01",
            "scope": "authentic source graph only",
            "source": {name: records[name] for name in GRAPH_HASHES},
            "measurement_alignment_verified": False,
            "measurement_quality_verified": False,
            "graph": report,
        }
        args.graph_report.write_text(
            json.dumps(report_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(f"Graph-only audit: {args.graph_report}; measurement alignment unverified")
    if not args.measurements:
        print(f"Measurements unavailable here. Download metr-la.h5 from {MEASUREMENT_URL}")
        print("Then rerun with --measurements /path/to/metr-la.h5")


if __name__ == "__main__":
    main()
