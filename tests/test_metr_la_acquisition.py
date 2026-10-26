"""Acquisition validates checksums and never overwrites differing source files."""

import hashlib
import runpy
from io import BytesIO
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_graph_acquisition_checksums_and_repeatability(tmp_path: Path, monkeypatch) -> None:
    acquire = runpy.run_path(str(REPO / "scripts/download_metrla.py"))["acquire_graph"]
    globals_ = acquire.__globals__
    assets = {"adj_mx.pkl": b"synthetic bytes only", "graph_sensor_ids.txt": b"SYN_01,SYN_02"}
    checksums = {name: hashlib.sha256(blob).hexdigest() for name, blob in assets.items()}
    monkeypatch.setitem(globals_, "GRAPH_HASHES", checksums)
    monkeypatch.setitem(
        globals_, "urlopen", lambda url, **kwargs: BytesIO(assets[url.rsplit("/", 1)[1]])
    )
    records = acquire(tmp_path)
    assert records["adj_mx.pkl"]["sha256"] == checksums["adj_mx.pkl"]
    monkeypatch.setitem(
        globals_, "urlopen", lambda *args, **kwargs: pytest.fail("Unexpected download")
    )
    assert acquire(tmp_path) == records
    path = tmp_path / "adj_mx.pkl"
    path.write_bytes(b"different existing source")
    with pytest.raises(ValueError, match="refusing overwrite"):
        acquire(tmp_path)
    assert path.read_bytes() == b"different existing source"


def test_graph_download_mismatched_hash_is_never_installed(tmp_path: Path, monkeypatch) -> None:
    acquire = runpy.run_path(str(REPO / "scripts/download_metrla.py"))["acquire_graph"]
    monkeypatch.setitem(acquire.__globals__, "urlopen", lambda *args, **kwargs: BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="Download checksum mismatch"):
        acquire(tmp_path)
    assert not (tmp_path / "adj_mx.pkl").exists()
