"""Execute the documented CLIs on labeled fixtures without external data or services."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from traffic_thi.io import sha256_file
from traffic_thi.metr_la_pipeline import run_e01

REPO = Path(__file__).resolve().parents[1]


def fixture_config(tmp_path: Path) -> Path:
    """Use committed source fixtures and isolated outputs for integration checks."""
    config = yaml.safe_load((REPO / "config/metr_la_synthetic.yaml").read_text())
    for name in ["measurements", "graph"]:
        config["paths"][name] = str(REPO / config["paths"][name])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_cli_prepare_audit_repeatability_and_provenance(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    environment = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "mpl"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }
    command = [
        sys.executable,
        str(REPO / "scripts/prepare_metr_la.py"),
        "--config",
        str(config),
        "--root",
        str(tmp_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, env=environment, check=True)
    assert "SYNTHETIC FIXTURE: shape=[40, 5], prepared=True" in completed.stdout
    result = tmp_path / "results/metrics/e01_synthetic_benchmark.json"
    manifest = json.loads(result.read_text())
    assert manifest["synthetic"] is True
    assert len(manifest["runs"]) == 40  # 2 protocols × 5 repeats × 4 rates
    assert manifest["partitions"]["train"]["count"] == 28
    assert manifest["partitions"]["validation"]["start_inclusive"] == 28
    assert manifest["partitions"]["test"]["count"] == 8
    assert len(manifest["sensor_ids"]) == 5
    audit_path = tmp_path / "results/metrics/e01_synthetic_audit.json"
    audit = json.loads(audit_path.read_text())
    assert audit["data"]["valid_entries"] == 195
    assert all(audit["checksum_verified_against_expected"].values())
    before = {
        str(path): sha256_file(path) for path in (tmp_path / "results").rglob("*") if path.is_file()
    }
    subprocess.run(command, capture_output=True, text=True, env=environment, check=True)
    assert before == {path: sha256_file(path) for path in before}
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/audit_metr_la.py"),
            "--config",
            str(config),
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )
    # Preparation never writes E00 outputs or modifies the fixture source.
    assert not list((tmp_path / "results").rglob("e00*"))
    assert manifest["source_sha256"] == yaml.safe_load(config.read_text())["expected_sha256"]
    for run in manifest["runs"]:
        assert run["observed_sensor_count"] + len(run["withheld_sensor_ids"]) == 5
        assert run["valid_heldout_entries"]["test"] > 0
        assert run["normalization"]["count"] <= 28 * 5


def test_checksum_failure_precedes_loading(tmp_path: Path) -> None:
    config_path = fixture_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["expected_sha256"]["measurements"] = "0" * 64
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run_e01(config_path, root=tmp_path, prepare=True)
    assert not (tmp_path / "results").exists()


def test_ambiguous_timestamps_audited_but_not_prepared(tmp_path: Path) -> None:
    config_path = fixture_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    frame = pd.read_csv(config["paths"]["measurements"])
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    local_dir = tmp_path / "source"
    local_dir.mkdir()
    path = local_dir / "duplicate.csv"
    frame.to_csv(path, index=False)
    config["paths"]["measurements"] = str(path)
    config["expected_sha256"]["measurements"] = sha256_file(path)
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Timestamps"):
        run_e01(config_path, root=tmp_path, prepare=True)
    audit = json.loads((tmp_path / "results/metrics/e01_synthetic_audit.json").read_text())
    assert audit["data"]["timestamps"]["duplicate_rows"] == 2
    assert audit["prepared"] is False
    assert not (tmp_path / "results/metrics/e01_synthetic_benchmark.json").exists()
