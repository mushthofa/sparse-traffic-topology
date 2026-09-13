"""Input and source-integrity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def load_bandung_google(path: str | Path) -> pd.DataFrame:
    """Load the immutable Google edge-sample table."""
    return pd.read_csv(path)


def load_sampled_edges(path: str | Path) -> pd.DataFrame:
    """Load sampled OSM edge metadata."""
    return pd.read_csv(path)


def sha256_file(path: str | Path) -> str:
    """Return a file's SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sha256_manifest(path: str | Path) -> dict[str, str]:
    """Load a standard SHA256SUMS manifest keyed by its relative paths."""
    entries: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        checksum, relative_path = line.split(maxsplit=1)
        entries[relative_path.removeprefix("*")] = checksum
    return entries


def bandung_integrity_report(
    google: pd.DataFrame,
    sampled: pd.DataFrame,
    *,
    expected_time_labels: list[str],
    raw_dir: Path,
) -> pd.DataFrame:
    """Return explicit schema, identity, coverage, consistency, and checksum checks."""
    edge_key = ["u", "v", "key"]
    observation_key = [*edge_key, "timestamp_label"]
    required_google = {
        *observation_key,
        "edge_id",
        "osm_length_m",
        "osm_time_s",
        "google_distance_m",
        "google_time_s",
        "status",
        "source",
    }
    required_sample = {*edge_key, "edge_id", "highway", "osm_length_m", "osm_time_s"}
    expected_ids = (
        google["u"].astype(str) + "_" + google["v"].astype(str) + "_" + google["key"].astype(str)
    )
    observed_pairs = google[observation_key].drop_duplicates()
    sampled_keys = sampled[edge_key].drop_duplicates()
    expected_pairs = len(sampled_keys) * len(expected_time_labels)
    google_keys = google[edge_key].drop_duplicates()
    merged = google_keys.merge(sampled_keys, on=edge_key, how="outer", indicator=True)

    records: list[dict[str, object]] = []

    def add(check: str, passed: bool, actual: object, expected: object) -> None:
        records.append(
            {"check": check, "passed": bool(passed), "actual": actual, "expected": expected}
        )

    add(
        "google_required_columns",
        required_google <= set(google),
        len(required_google & set(google)),
        len(required_google),
    )
    add(
        "sample_required_columns",
        required_sample <= set(sampled),
        len(required_sample & set(sampled)),
        len(required_sample),
    )
    add(
        "edge_id_matches_u_v_key",
        expected_ids.eq(google["edge_id"].astype(str)).all(),
        int(expected_ids.eq(google["edge_id"].astype(str)).sum()),
        len(google),
    )
    add(
        "unique_directed_edge_time",
        not google.duplicated(observation_key).any(),
        int(observed_pairs.shape[0]),
        len(google),
    )
    add(
        "complete_edge_time_grid",
        len(observed_pairs) == expected_pairs,
        len(observed_pairs),
        expected_pairs,
    )
    add(
        "configured_time_labels",
        set(google["timestamp_label"]) == set(expected_time_labels),
        google["timestamp_label"].nunique(),
        len(expected_time_labels),
    )
    add(
        "google_edges_match_sample",
        merged["_merge"].eq("both").all(),
        int(merged["_merge"].eq("both").sum()),
        len(merged),
    )
    add(
        "source_is_google",
        google["source"].eq("google").all(),
        int(google["source"].eq("google").sum()),
        len(google),
    )

    manifest = load_sha256_manifest(raw_dir / "SHA256SUMS.txt")
    for relative_path, expected_hash in sorted(manifest.items()):
        source_path = raw_dir.parents[2] / relative_path
        actual_hash = sha256_file(source_path)
        add(
            f"sha256:{Path(relative_path).name}",
            actual_hash == expected_hash,
            actual_hash,
            expected_hash,
        )
    return pd.DataFrame.from_records(records)
