"""Synthetic-only ingestion and graph validation tests; no scientific METR-LA claims."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_thi.metr_la import (
    graph_audit,
    load_measurements,
    load_sensor_graph,
    measurement_audit,
    sensor_ids,
    timestamp_audit,
    valid_speeds,
    validate_timestamps,
)

FIXTURES = Path(__file__).parent / "fixtures/metr_la"


def test_csv_ingestion_retains_source_values_order_and_timestamps() -> None:
    frame = load_measurements(FIXTURES / "speeds.csv")
    assert frame.shape == (40, 5)
    assert list(frame.columns) == [f"SYN_{i:02d}" for i in range(1, 6)]
    assert frame.index[0] == pd.Timestamp("2012-01-01 23:00")
    assert np.isnan(frame.iloc[0, 0])
    assert frame.iloc[1, 1] == 0
    assert frame.iloc[2, 2] == -1
    assert np.isposinf(frame.iloc[3, 3])
    assert np.isneginf(frame.iloc[4, 4])


def test_hdf_roundtrip_preserves_exact_order_timezone_and_units(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [[5.0, 12.0], [7.0, 14.0]],
        columns=["001", "2"],
        index=pd.date_range("2012-06-01", periods=2, freq="5min", tz="UTC"),
    )
    path = tmp_path / "local.h5"
    frame.to_hdf(path, key="df", format="table")
    pd.testing.assert_frame_equal(load_measurements(path, hdf_key="df"), frame, check_freq=False)


@pytest.mark.parametrize("labels", [["a", "a"], [""], [" a"], [None], [float("nan")], [pd.NA]])
def test_invalid_ids_fail(labels: list) -> None:
    with pytest.raises(ValueError, match="sensor|Sensor"):
        sensor_ids(labels)


def test_csv_leading_zero_ids_and_duplicate_headers(tmp_path: Path) -> None:
    path = tmp_path / "speeds.csv"
    path.write_text("timestamp,001,2\n2012-01-01,10,20\n")
    assert list(load_measurements(path)) == ["001", "2"]
    path.write_text("timestamp,001,001\n2012-01-01,10,20\n")
    with pytest.raises(ValueError, match="Duplicated"):
        load_measurements(path)


def test_timestamp_audit_counts_without_repairing() -> None:
    index = pd.DatetimeIndex(
        ["2012-01-01 00:00", "2012-01-01 00:05", "2012-01-01 00:05", "2012-01-01 00:17", None]
    )
    report = timestamp_audit(index)
    assert report["missing_timestamp_count"] == 2
    assert report["duplicate_rows"] == 2
    assert report["duplicate_excess"] == 1
    assert report["nat_count"] == 1
    assert report["off_grid_count"] == 1
    assert report["nonstandard_intervals"] == 2
    with pytest.raises(ValueError, match="Timestamps"):
        validate_timestamps(index)
    with pytest.raises(ValueError, match="Timestamps"):
        validate_timestamps(pd.date_range("2012", periods=4, freq="5min")[::-1])


def test_zero_missingness_constant_and_graph_audit() -> None:
    frame = load_measurements(FIXTURES / "speeds.csv")
    report, tables = measurement_audit(frame)
    assert report["valid_entries"] == 195
    assert report["nan_count"] == 1
    assert report["infinity_count"] == 2
    assert report["zero_count"] == 1
    assert report["negative_finite_count"] == 1
    assert report["timestamps"]["missing_timestamp_count"] == 1
    sensors = tables["sensors"].set_index("sensor_id")
    assert sensors.loc["SYN_05", "constant"]
    assert sensors.loc["SYN_04", "nearly_constant"]
    assert tables["hour"]["total_entries"].sum() == 200
    assert tables["day_of_week"]["valid_entries"].sum() == 195
    graph = load_sensor_graph(FIXTURES / "adjacency.csv", frame.columns)
    stats, degrees, edges = graph_audit(graph)
    assert stats["directed_edges_including_loops"] == 5
    assert stats["directed_edges_excluding_loops"] == 4
    assert stats["undirected_edges_excluding_loops"] == 3
    assert stats["source_self_loops"] == 1
    assert stats["isolated_nodes_ignoring_loops"] == 1
    assert stats["weak_components"] == 2
    assert graph.undirected()["SYN_01"]["SYN_02"]["weight"] == 0.8
    assert graph.directed()["SYN_02"]["SYN_01"]["weight"] == 0.4
    assert len(degrees) == 5 and len(edges) == 5


def test_empty_valid_sensor_and_zero_policy() -> None:
    values = np.array([[np.nan, np.inf, -np.inf, -1, 0, 5]])
    assert valid_speeds(values).tolist() == [[False, False, False, False, False, True]]
    assert valid_speeds(values, zero_policy="include")[0, 4]
    with pytest.raises(ValueError, match="zero_policy"):
        valid_speeds(values, zero_policy="invent")
    frame = pd.DataFrame([[np.nan], [0]], index=pd.date_range("2012", periods=2, freq="5min"))
    _, tables = measurement_audit(frame)
    assert tables["sensors"].iloc[0]["valid_count"] == 0
    assert not tables["sensors"].iloc[0]["constant"]
    assert np.isnan(tables["sensors"].iloc[0]["mean"])


def test_graph_order_permutation_is_explicit_and_two_axis(tmp_path: Path) -> None:
    ids = ["001", "2", "3"]
    matrix = np.array([[1.0, 0.2, 0.3], [0.4, 1, 0], [0, 0.8, 1]])
    path = tmp_path / "graph.csv"
    pd.DataFrame(matrix, index=ids, columns=ids).to_csv(path)
    with pytest.raises(ValueError, match="order differs"):
        load_sensor_graph(path, ["3", "001", "2"])
    graph = load_sensor_graph(path, ["3", "001", "2"], allow_permutation=True)
    assert graph.permutation == (2, 0, 1)
    np.testing.assert_array_equal(graph.adjacency, matrix[np.ix_([2, 0, 1], [2, 0, 1])])
    assert graph.source_ids == tuple(ids)
    with pytest.raises(ValueError, match="do not match"):
        load_sensor_graph(path, ["001", "2", "missing"], allow_permutation=True)
    pd.DataFrame(matrix, index=["2", "001", "3"], columns=ids).to_csv(path)
    with pytest.raises(ValueError, match="row and column"):
        load_sensor_graph(path, ids)


@pytest.mark.parametrize("fault", ["mapping", "ids", "shape", "negative", "nonfinite", "duplicate"])
def test_graph_pickle_inconsistent_inputs_fail(tmp_path: Path, fault: str) -> None:
    ids, mapping, matrix = ["a", "b"], {"a": 0, "b": 1}, np.eye(2)
    if fault == "mapping":
        mapping = {"a": 1, "b": 0}
    elif fault == "ids":
        ids, mapping = ["a", "x"], {"a": 0, "x": 1}
    elif fault == "shape":
        matrix = np.eye(3)
    elif fault == "negative":
        matrix[0, 1] = -1
    elif fault == "nonfinite":
        matrix[0, 1] = np.nan
    else:
        ids = ["a", "a"]
    path = tmp_path / "graph.pkl"
    path.write_bytes(pickle.dumps([ids, mapping, matrix], protocol=2))
    with pytest.raises(ValueError):
        load_sensor_graph(path, ["a", "b"], trust_pickle=True)


def test_graph_pickle_requires_trust_and_preserves_direction(tmp_path: Path) -> None:
    path = tmp_path / "graph.pkl"
    matrix = np.array([[1, 0.3], [0, 1]])
    path.write_bytes(pickle.dumps([["a", "b"], {"a": 0, "b": 1}, matrix], protocol=2))
    with pytest.raises(ValueError, match="trust_pickle"):
        load_sensor_graph(path, ["a", "b"])
    graph = load_sensor_graph(path, ["a", "b"], trust_pickle=True)
    np.testing.assert_array_equal(graph.adjacency, matrix)
    assert not graph.directed().has_edge("b", "a")
    with pytest.raises(ValueError, match="not distances"):
        load_sensor_graph(path, ["a", "b"], weight_kind="distance")
