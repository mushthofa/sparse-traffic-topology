from pathlib import Path

import pandas as pd

from traffic_thi.io import bandung_integrity_report, load_sha256_manifest, sha256_file


def test_checksum_helpers(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    checksum = sha256_file(source)
    manifest = tmp_path / "SHA256SUMS.txt"
    manifest.write_text(f"{checksum}  source.csv\n", encoding="utf-8")
    assert load_sha256_manifest(manifest) == {"source.csv": checksum}


def test_bandung_integrity_report(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data/raw/bandung"
    raw_dir.mkdir(parents=True)
    (raw_dir / "SHA256SUMS.txt").write_text("", encoding="utf-8")
    google = pd.DataFrame(
        {
            "edge_id": ["1_2_0"],
            "u": [1],
            "v": [2],
            "key": [0],
            "timestamp_label": ["mon_01"],
            "osm_length_m": [10.0],
            "osm_time_s": [1.0],
            "google_distance_m": [10.0],
            "google_time_s": [2.0],
            "status": ["ok"],
            "source": ["google"],
        }
    )
    sampled = pd.DataFrame(
        {
            "edge_id": ["1_2_0"],
            "u": [1],
            "v": [2],
            "key": [0],
            "highway": ["residential"],
            "osm_length_m": [10.0],
            "osm_time_s": [1.0],
        }
    )
    report = bandung_integrity_report(
        google, sampled, expected_time_labels=["mon_01"], raw_dir=raw_dir
    )
    assert report["passed"].all()
    assert "complete_edge_time_grid" in set(report["check"])
