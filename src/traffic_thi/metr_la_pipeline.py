"""Reproducible E01 orchestration; no reconstruction models or performance claims."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure

from traffic_thi.io import sha256_file
from traffic_thi.metr_la import (
    graph_audit,
    load_measurements,
    load_sensor_graph,
    measurement_audit,
    valid_speeds,
    validate_timestamps,
)
from traffic_thi.splits import (
    nested_sensor_masks,
    protocol_masks,
    temporal_partitions,
    training_normalization,
)


def write_json(path: Path, value: object) -> None:
    """Write strict, stable JSON (undefined metrics must be null, never NaN)."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _save_svg(figure: Figure, path: Path) -> None:
    # Keep labels as reviewable SVG text, remove variable dates and whitespace.
    with matplotlib.rc_context({"svg.hashsalt": "traffic-thi-e01", "svg.fonttype": "none"}):
        figure.savefig(path, metadata={"Date": None})
    plt.close(figure)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def _figure(path: Path, values: np.ndarray, *, xlabel: str, title: str) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.hist(values, bins=40, color="#2878B5", edgecolor="white")
    axis.set(xlabel=xlabel, ylabel="Entries", title=title)
    fig.tight_layout()
    _save_svg(fig, path)


def run_e01(
    config_path: str | Path,
    *,
    root: str | Path,
    prepare: bool = False,
    trust_graph_pickle: bool = False,
    allow_graph_permutation: bool = False,
) -> dict:
    """Audit local sources; optionally save separate target/validity arrays and splits.

    Outputs are prefixed e01 or e01_synthetic; no E00 artifact is touched. Full
    targets are evaluator-only. Model inputs must be obtained via protocol_masks
    and permitted_measurements, never passed the evaluator archive directly.
    """
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    source = config["source"]
    if source["kind"] not in {"real", "synthetic"}:
        raise ValueError("source.kind must explicitly be real or synthetic")
    for name in ["version", "measurement_url", "graph_url", "timezone_assumption", "unit_basis"]:
        if not source.get(name):
            raise ValueError(f"Missing provenance field: {name}")
    if source["speed_units"] != "mph":
        raise ValueError(
            "This benchmark requires speeds declared in mph; no conversion is implicit"
        )
    prefix = "e01_synthetic" if source["kind"] == "synthetic" else "e01"
    paths = {key: (root / config["paths"][key]).resolve() for key in ["measurements", "graph"]}
    # Hash before any potentially pickle-backed deserialization.
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    expected = config.get("expected_sha256", {})
    for key, digest in hashes.items():
        if expected.get(key) and expected[key] != digest:
            raise ValueError(f"SHA256 mismatch for {key}")
    frame = load_measurements(paths["measurements"], hdf_key=config["paths"].get("hdf_key"))
    graph = load_sensor_graph(
        paths["graph"],
        frame.columns,
        trust_pickle=trust_graph_pickle,
        allow_permutation=allow_graph_permutation,
        weight_kind=source["graph_weight_kind"],
    )
    qc = config["quality"]
    data_report, tables = measurement_audit(
        frame,
        zero_policy=qc["zero_policy"],
        near_constant_cv=qc["near_constant_cv"],
        frequency=qc["sampling_interval"],
    )
    graph_report, degrees, edges = graph_audit(graph)
    metrics_dir, table_dir, figure_dir = [
        root / "results" / part for part in ["metrics", "tables", "figures"]
    ]
    processed = (root / config["paths"]["processed"]).resolve()
    # Never write into immutable source directories, including symlink aliases.
    outputs = [metrics_dir.resolve(), table_dir.resolve(), figure_dir.resolve(), processed]
    protected = [(root / "data/raw").resolve(), *(path.parent for path in paths.values())]
    if any(output == path or path in output.parents for output in outputs for path in protected):
        raise ValueError("Outputs must be outside raw/source directories")
    for path in outputs[:3]:
        path.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment_id": "E01",
        "source": source,
        "synthetic": source["kind"] == "synthetic",
        "source_sha256": hashes,
        "checksum_verified_against_expected": {key: bool(expected.get(key)) for key in paths},
        "source_files": {
            key: str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            for key, path in paths.items()
        },
        "config_sha256": sha256_file(config_path),
        "configuration": config,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "networkx": nx.__version__,
        },
        "data": data_report,
        "graph": graph_report,
        "expected_shape": config["expectations"]["shape"],
        "expected_shape_matches": data_report["shape"] == config["expectations"]["shape"],
        "alignment": {
            "measurement_ids": list(graph.ids),
            "source_graph_ids": list(graph.source_ids),
            "permutation_source_indices": list(graph.permutation),
            "explicit_permutation_allowed": allow_graph_permutation,
        },
        "prepared": False,
    }
    write_json(metrics_dir / f"{prefix}_audit.json", report)
    for name, table in {**tables, "graph_degrees": degrees, "directed_edges": edges}.items():
        table.to_csv(table_dir / f"{prefix}_{name}.csv", index=False)
    undirected_edges = pd.DataFrame(
        [(u, v, d["weight"]) for u, v, d in graph.undirected().edges(data=True)],
        columns=["sensor_id_1", "sensor_id_2", "max_affinity"],
    )
    undirected_edges.to_csv(table_dir / f"{prefix}_undirected_edges.csv", index=False)
    label = "SYNTHETIC FIXTURE — " if report["synthetic"] else "METR-LA — "
    values = frame.to_numpy()
    valid = valid_speeds(values, zero_policy=qc["zero_policy"])
    _figure(
        figure_dir / f"{prefix}_speed.svg",
        values[valid],
        xlabel="Valid speed (mph)",
        title=label + "speed distribution",
    )
    _figure(
        figure_dir / f"{prefix}_affinity.svg",
        edges["affinity"].to_numpy(),
        xlabel="Directed nonzero affinity (includes source self-loops)",
        title=label + "source graph weights",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, name, xlabel in zip(
        axes,
        ["hour", "day_of_week"],
        ["Hour (source clock)", "Day of week (Monday=0)"],
        strict=True,
    ):
        axis.plot(tables[name][name], tables[name]["mean"], marker="o")
        axis.set(xlabel=xlabel, ylabel="Mean valid speed (mph)")
    fig.suptitle(label + "temporal patterns")
    fig.tight_layout()
    _save_svg(fig, figure_dir / f"{prefix}_temporal.svg")
    if not prepare:
        return report
    # Persist an audit before failing on ambiguous timestamps; never repair them silently.
    validate_timestamps(frame.index)
    benchmark = config["benchmark"]
    partitions = temporal_partitions(frame.index, tuple(benchmark["temporal_rates"]))
    masks = nested_sensor_masks(
        frame.columns,
        fractions=tuple(benchmark["observation_fractions"]),
        seeds=tuple(benchmark["seeds"]),
    )
    manifest = {
        "experiment_id": "E01",
        "synthetic": report["synthetic"],
        "source_sha256": hashes,
        "config_sha256": sha256_file(config_path),
        "zero_policy": qc["zero_policy"],
        "sensor_ids": list(graph.ids),
        "static_graph_available": True,
        "mask_method": "SHA256 of compact UTF-8 JSON [seed,sensor_id], byte ranking; floor(f*N)",
        "temporal_rounding": "floor train, floor validation, remainder test",
        "partitions": {
            name: {
                "start_inclusive": section.start,
                "stop_exclusive": section.stop,
                "count": section.stop - section.start,
                "first_timestamp": frame.index[section.start].isoformat(),
                "last_timestamp": frame.index[section.stop - 1].isoformat(),
            }
            for name, section in partitions.items()
        },
        "runs": [],
    }
    for mask in masks:
        observed = mask.observed(frame.columns)
        for protocol in ["A", "B"]:
            inputs, evaluation = protocol_masks(valid, observed, partitions, protocol=protocol)
            manifest["runs"].append(
                {
                    "protocol": protocol,
                    "repetition": mask.repetition,
                    "seed": mask.seed,
                    "requested_observation_fraction": mask.fraction,
                    "observed_sensor_count": int(observed.sum()),
                    "actual_observation_fraction": float(observed.mean()),
                    "observed_sensor_ids": sorted(mask.selected_ids),
                    "withheld_sensor_ids": sorted(set(graph.ids) - set(mask.selected_ids)),
                    "normalization": training_normalization(values, inputs, partitions["train"]),
                    "permitted_entries": {
                        name: int(inputs[section].sum()) for name, section in partitions.items()
                    },
                    "valid_heldout_entries": {
                        name: int(mask.sum()) for name, mask in evaluation.items()
                    },
                }
            )
    processed.mkdir(parents=True, exist_ok=True)
    # Local-only compressed evaluator archive. Do not give this complete archive to a model.
    np.savez_compressed(
        processed / f"{prefix}_evaluator_targets.npz",
        targets=values,
        valid=valid,
        sensor_ids=np.array(graph.ids),
        timestamps=np.array(frame.index.astype(str)),
    )
    pd.DataFrame({"row": np.arange(len(frame)), "timestamp": frame.index.astype(str)}).to_csv(
        table_dir / f"{prefix}_timestamps.csv", index=False
    )
    write_json(metrics_dir / f"{prefix}_benchmark.json", manifest)
    report["prepared"] = True
    write_json(metrics_dir / f"{prefix}_audit.json", report)
    return report
