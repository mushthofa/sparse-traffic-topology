"""METR-LA source ingestion, sensor-graph alignment, and descriptive auditing.

No interpolation, graph fitting, unit conversion, or timestamp sorting occurs here.
"""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def sensor_ids(values: object) -> tuple[str, ...]:
    """Represent identifiers as exact strings; reject ambiguous or duplicate labels."""
    values = list(values)
    if any(pd.isna(v) for v in values):
        raise ValueError("Sensor IDs must not be missing")
    ids = tuple(v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values)
    if not ids or any(not v or v != v.strip() or v.lower() in {"nan", "none"} for v in ids):
        raise ValueError("Sensor IDs must be nonempty, nonmissing, whitespace-free strings")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicated sensor IDs")
    return ids


def _csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return next(csv.reader(stream))


def load_measurements(path: str | Path, *, hdf_key: str | None = None) -> pd.DataFrame:
    """Read a DCRNN HDF DataFrame or timestamp-first wide CSV, preserving order.

    HDF files must be trusted local files: PyTables fixed stores can contain pickles.
    CSV is also supported for traceable local conversions and synthetic fixtures.
    """
    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        frame = pd.read_hdf(path, key=hdf_key)
    elif path.suffix.lower() == ".csv":
        ids = sensor_ids(_csv_header(path)[1:])
        frame = pd.read_csv(path, index_col=0)
        frame.columns = ids
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    else:
        raise ValueError("Measurements must be DCRNN HDF5 or a timestamp-first wide CSV")
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Measurements require a DataFrame with a DatetimeIndex")
    frame = frame.copy()
    frame.columns = sensor_ids(frame.columns)
    if frame.empty:
        raise ValueError("Measurements must contain timestamps and sensors")
    try:
        return frame.astype(float)
    except (ValueError, TypeError) as error:
        raise ValueError("Speed readings must be numeric (missing values may be NaN)") from error


def timestamp_audit(index: pd.DatetimeIndex, *, frequency: str = "5min") -> dict:
    """Report source-order timestamp defects without dropping or inserting rows."""
    step = pd.Timedelta(frequency)
    if step <= pd.Timedelta(0):
        raise ValueError("Sampling interval must be positive")
    clean = index[~index.isna()]
    unique = clean.unique().sort_values()
    differences = clean[1:] - clean[:-1]
    grid = pd.date_range(unique.min(), unique.max(), freq=step) if len(unique) else unique
    missing = grid.difference(unique)
    return {
        "start": str(clean.min()) if len(clean) else None,
        "end": str(clean.max()) if len(clean) else None,
        "timezone": str(index.tz) if index.tz is not None else "naive (no offset encoded)",
        "nat_count": int(index.isna().sum()),
        "duplicate_rows": int(index.duplicated(keep=False).sum()),
        "duplicate_excess": int(index.duplicated().sum()),
        "monotonic_increasing": bool(index.is_monotonic_increasing),
        "missing_timestamp_count": len(missing),
        "missing_timestamps": [t.isoformat() for t in missing],
        "off_grid_count": int((~unique.isin(grid)).sum()),
        "nonstandard_intervals": int((differences != step).sum()),
        "interval_counts_seconds": {
            str(float(k.total_seconds())): int(v)
            for k, v in pd.Series(differences).value_counts().sort_index().items()
        },
        "expected_interval_seconds": step.total_seconds(),
    }


def validate_timestamps(index: pd.DatetimeIndex) -> None:
    """Require unambiguous chronological partitions; gaps remain explicit in audits."""
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Timestamps must be nonmissing, unique, and chronological; see audit")


def valid_speeds(values: np.ndarray, *, zero_policy: str = "exclude") -> np.ndarray:
    """Exclude NaN, infinity, negatives, and (by default) ambiguous zero readings."""
    if zero_policy not in {"exclude", "include"}:
        raise ValueError("zero_policy must be exclude or include")
    values = np.asarray(values, dtype=float)
    return np.isfinite(values) & (values >= 0 if zero_policy == "include" else values > 0)


@dataclass(frozen=True)
class SensorGraph:
    """Adjacency in measurement order, plus its explicit source-order permutation."""

    ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    adjacency: np.ndarray
    permutation: tuple[int, ...]
    weight_kind: str

    def directed(self) -> nx.DiGraph:
        """Keep every nonzero directed weight, including source self-loops."""
        graph = nx.from_numpy_array(self.adjacency, create_using=nx.DiGraph)
        return nx.relabel_nodes(graph, dict(enumerate(self.ids)))

    def undirected(self) -> nx.Graph:
        """Union directions using max affinity; exclude self-loops explicitly."""
        matrix = np.maximum(self.adjacency, self.adjacency.T).copy()
        np.fill_diagonal(matrix, 0)
        graph = nx.from_numpy_array(matrix, create_using=nx.Graph)
        return nx.relabel_nodes(graph, dict(enumerate(self.ids)))


def load_sensor_graph(
    path: str | Path,
    measurement_ids: object,
    *,
    allow_permutation: bool = False,
    trust_pickle: bool = False,
    weight_kind: str = "affinity",
) -> SensorGraph:
    """Read DCRNN [IDs, ID-to-index, adjacency] pickle or labeled matrix CSV.

    Different orders fail unless allow_permutation is explicit. In that case both
    adjacency axes are permuted together and the exact permutation is recorded.
    Only nonnegative affinities are supported by the max-affinity symmetrization.
    """
    path = Path(path)
    target = sensor_ids(measurement_ids)
    if weight_kind != "affinity":
        raise ValueError("This graph pipeline requires documented affinity weights, not distances")
    if path.suffix.lower() in {".pkl", ".pickle"}:
        if not trust_pickle:
            raise ValueError("Graph pickle requires explicit trust_pickle for a trusted source")
        with path.open("rb") as stream:
            source, mapping, matrix = pickle.load(stream, encoding="latin1")
        source = sensor_ids(source)
        normalized_mapping = {sensor_ids([key])[0]: value for key, value in mapping.items()}
        if len(normalized_mapping) != len(mapping) or normalized_mapping != {
            key: i for i, key in enumerate(source)
        }:
            raise ValueError("Graph ID-to-index mapping is inconsistent with source sensor order")
    elif path.suffix.lower() == ".csv":
        source = sensor_ids(_csv_header(path)[1:])
        frame = pd.read_csv(path, index_col=0, dtype=str, keep_default_na=False)
        # Read index explicitly as text: pandas otherwise erases leading zeroes.
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))[1:]
        row_ids = sensor_ids([row[0] for row in rows])
        if row_ids != source:
            raise ValueError("Graph row and column sensor IDs have inconsistent order")
        matrix = frame.to_numpy(dtype=float)
    else:
        raise ValueError("Graph must be a DCRNN pickle or labeled adjacency CSV")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (len(source), len(source)):
        raise ValueError("Graph adjacency shape does not match its sensor IDs")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Graph affinities must be finite and nonnegative")
    if set(source) != set(target):
        raise ValueError("Graph sensor IDs do not match measurement sensor IDs")
    if source != target and not allow_permutation:
        raise ValueError("Graph sensor order differs; explicitly allow and record permutation")
    positions = {key: i for i, key in enumerate(source)}
    permutation = tuple(positions[key] for key in target)
    aligned = matrix[np.ix_(permutation, permutation)].copy()
    return SensorGraph(target, source, aligned, permutation, weight_kind)


def _stats(values: object) -> dict:
    series = pd.Series(
        values if isinstance(values, (np.ndarray, pd.Series)) else list(values), dtype=float
    )
    return {
        key: (float(value) if pd.notna(value) else None) for key, value in series.describe().items()
    }


def graph_audit(graph: SensorGraph) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Describe source-directed and derived-undirected graphs separately."""
    directed, undirected = graph.directed(), graph.undirected()
    degrees = pd.DataFrame(
        {
            "sensor_id": graph.ids,
            "in_degree": [directed.in_degree(n) for n in graph.ids],
            "out_degree": [directed.out_degree(n) for n in graph.ids],
            "undirected_degree": [undirected.degree(n) for n in graph.ids],
            "in_strength": [directed.in_degree(n, weight="weight") for n in graph.ids],
            "out_strength": [directed.out_degree(n, weight="weight") for n in graph.ids],
            "weak_component": [
                next(i for i, c in enumerate(nx.weakly_connected_components(directed)) if n in c)
                for n in graph.ids
            ],
            "undirected_component": [
                next(i for i, c in enumerate(nx.connected_components(undirected)) if n in c)
                for n in graph.ids
            ],
        }
    )
    edges = pd.DataFrame(
        [(u, v, d["weight"]) for u, v, d in directed.edges(data=True)],
        columns=["source_sensor_id", "target_sensor_id", "affinity"],
    )
    loopless = directed.copy()
    loopless.remove_edges_from(nx.selfloop_edges(loopless))
    report = {
        "nodes": len(graph.ids),
        "directed_edges_including_loops": directed.number_of_edges(),
        "source_self_loops": nx.number_of_selfloops(directed),
        "directed_edges_excluding_loops": loopless.number_of_edges(),
        "undirected_edges_excluding_loops": undirected.number_of_edges(),
        "weak_components": nx.number_weakly_connected_components(directed),
        "strong_components": nx.number_strongly_connected_components(directed),
        "undirected_components": nx.number_connected_components(undirected),
        "isolated_nodes_ignoring_loops": len(list(nx.isolates(undirected))),
        "directed_density_excluding_loops": nx.density(loopless),
        "undirected_density_excluding_loops": nx.density(undirected),
        "weight_kind": graph.weight_kind,
        "directed_affinity_statistics": _stats(edges["affinity"]),
        "undirected_affinity_statistics": _stats(
            d["weight"] for _, _, d in undirected.edges(data=True)
        ),
        "degree_statistics": {
            column: _stats(degrees[column])
            for column in ["in_degree", "out_degree", "undirected_degree"]
        },
        "symmetrization": "max(A, A.T), diagonal zero; no distance conversion",
    }
    return report, degrees, edges


def measurement_audit(
    frame: pd.DataFrame,
    *,
    zero_policy: str = "exclude",
    near_constant_cv: float = 0.01,
    frequency: str = "5min",
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Audit all rows; descriptive statistics use only policy-valid source speeds."""
    if not np.isfinite(near_constant_cv) or near_constant_cv < 0:
        raise ValueError("near_constant_cv must be finite and nonnegative")
    values = frame.to_numpy(dtype=float)
    valid = valid_speeds(values, zero_policy=zero_policy)
    clean = frame.where(valid)
    count = clean.count()
    mean, std = clean.mean(), clean.std(ddof=0)
    cv = std / mean.replace(0, np.nan)
    sensors = pd.DataFrame(
        {
            "sensor_id": frame.columns,
            "valid_count": count.to_numpy(),
            "completeness": count.to_numpy() / len(frame),
            "mean": mean.to_numpy(),
            "median": clean.median().to_numpy(),
            "standard_deviation_ddof0": std.to_numpy(),
            "coefficient_of_variation": cv.to_numpy(),
            "constant": ((count > 0) & (clean.nunique() == 1)).to_numpy(),
            "nearly_constant": ((count >= 2) & (cv <= near_constant_cv)).to_numpy(),
            "nan_count": np.isnan(values).sum(axis=0),
            "infinity_count": np.isinf(values).sum(axis=0),
            "zero_count": (values == 0).sum(axis=0),
            "negative_finite_count": (np.isfinite(values) & (values < 0)).sum(axis=0),
        }
    )
    # Bounded long representation for temporal summaries, retaining every invalid entry as NaN.
    long = pd.DataFrame(
        {
            "speed": clean.to_numpy().ravel(),
            "hour": np.repeat(frame.index.hour, len(frame.columns)),
            "day_of_week": np.repeat(frame.index.day_of_week, len(frame.columns)),
        }
    )
    tables = {"sensors": sensors}
    for name, columns in [
        ("hour", ["hour"]),
        ("day_of_week", ["day_of_week"]),
        ("hour_of_week", ["day_of_week", "hour"]),
    ]:
        tables[name] = (
            long.groupby(columns, dropna=False)["speed"]
            .agg(
                total_entries="size",
                valid_entries="count",
                mean="mean",
                median="median",
                standard_deviation="std",
            )
            .reset_index()
        )
    report = {
        "shape": list(values.shape),
        "timestamps": timestamp_audit(frame.index, frequency=frequency),
        "total_entries": values.size,
        "valid_entries": int(valid.sum()),
        "invalid_entries": int((~valid).sum()),
        "completeness": float(valid.mean()),
        "nan_count": int(np.isnan(values).sum()),
        "infinity_count": int(np.isinf(values).sum()),
        "zero_count": int((values == 0).sum()),
        "negative_finite_count": int((np.isfinite(values) & (values < 0)).sum()),
        "zero_policy": zero_policy,
        "near_constant_cv_threshold": near_constant_cv,
        "network_valid_speed_statistics": _stats(values[valid]),
    }
    return report, tables
