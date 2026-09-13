"""Generate the reproducible E00 Bandung data-audit artifacts."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from traffic_thi.io import bandung_integrity_report, load_bandung_google, load_sampled_edges
from traffic_thi.quality import (
    exclusion_counts,
    qc_sensitivity,
    road_class_representativeness,
    temporal_variability,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.yaml")
    return parser.parse_args()


def save_distribution(series: pd.Series, path: Path, xlabel: str, title: str) -> None:
    """Save a deterministic histogram for finite values in ``series``."""
    plt.rcParams["svg.hashsalt"] = "traffic-thi-e00"
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    series.dropna().plot.hist(bins=40, ax=ax, color="#2878B5", edgecolor="white")
    ax.set(xlabel=xlabel, ylabel="Observations", title=title)
    fig.tight_layout()
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def save_temporal_variability(report: pd.DataFrame, path: Path) -> None:
    """Save median and 5–95% log-stress bands in configured temporal order."""
    plt.rcParams["svg.hashsalt"] = "traffic-thi-e00-temporal"
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    positions = range(len(report))
    ax.plot(positions, report["median"], marker="o", label="Median")
    ax.fill_between(positions, report["q05"], report["q95"], alpha=0.25, label="5–95%")
    ax.set_xticks(list(positions), report["timestamp_label"], rotation=60, ha="right")
    ax.set(ylabel="Included log stress", xlabel="Time state", title="Temporal variability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def save_road_class_comparison(report: pd.DataFrame, path: Path) -> None:
    """Save sampled versus full-network directed-edge class shares."""
    plt.rcParams["svg.hashsalt"] = "traffic-thi-e00-road-class"
    plot = report.set_index("highway")[["population_share", "sample_share"]]
    ax = plot.plot.bar(figsize=(9.0, 4.8), color=["#999999", "#2878B5"])
    ax.set(ylabel="Directed-edge share", xlabel="Highway class", title="Sample representativeness")
    ax.legend(["Full network", "Google sample"])
    ax.figure.tight_layout()
    ax.figure.savefig(path, metadata={"Date": None})
    plt.close(ax.figure)


def main() -> None:
    """Run E00 without modifying the immutable source snapshot."""
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    bandung = config["bandung"]
    qc = bandung["qc"]
    source = ROOT / config["bandung"]["google_samples"]
    frame = load_bandung_google(source)
    sampled = load_sampled_edges(ROOT / bandung["sampled_edges"])
    integrity = bandung_integrity_report(
        frame,
        sampled,
        expected_time_labels=bandung["time_labels"],
        raw_dir=ROOT / "data/raw/bandung",
    )
    if not integrity["passed"].all():
        failures = integrity.loc[~integrity["passed"], "check"].tolist()
        raise ValueError(f"Bandung source integrity checks failed: {failures}")
    audited_profiles, sensitivity = qc_sensitivity(
        frame, qc["profiles"], remove_self_loops=qc["remove_self_loops"]
    )
    primary_profile = qc["primary_profile"]
    audited = audited_profiles[primary_profile]

    metric_dir = ROOT / "results/metrics"
    table_dir = ROOT / "results/tables"
    figure_dir = ROOT / "results/figures"
    processed_dir = ROOT / "data/processed"
    for directory in (metric_dir, table_dir, figure_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    reason_table = exclusion_counts(audited)
    reason_table.to_csv(table_dir / "e00_exclusion_report.csv", index=False)
    sensitivity.to_csv(table_dir / "e00_qc_sensitivity.csv", index=False)
    integrity.to_csv(table_dir / "e00_integrity_checks.csv", index=False)
    distributions = audited[["route_ratio", "route_confidence", "log_stress"]].describe(
        percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
    )
    distributions.rename_axis("statistic").to_csv(table_dir / "e00_distribution_summary.csv")
    coverage = (
        audited.groupby(["timestamp_label", "day_name", "period"], sort=False)
        .agg(
            observations=("edge_id", "size"),
            unique_directed_edges=("edge_id", "nunique"),
            included_observations=("qc_included", "sum"),
        )
        .reset_index()
    )
    coverage.to_csv(table_dir / "e00_temporal_coverage.csv", index=False)
    temporal = temporal_variability(audited)
    temporal["_order"] = temporal["timestamp_label"].map(
        {label: position for position, label in enumerate(bandung["time_labels"])}
    )
    temporal = temporal.sort_values("_order").drop(columns="_order")
    temporal.to_csv(table_dir / "e00_temporal_variability.csv", index=False)
    population = pd.read_csv(
        ROOT / bandung["full_dynamic_gz"], usecols=["u", "v", "key", "highway"]
    ).drop_duplicates(["u", "v", "key"])
    representativeness = road_class_representativeness(sampled, population)
    representativeness.to_csv(table_dir / "e00_road_class_representativeness.csv", index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "observations_total",
                "observations_included",
                "observations_excluded",
                "unique_directed_edges",
                "time_states",
                "self_loop_observations",
                "duplicate_edge_time_observations",
                "status_ok_observations",
            ],
            "value": [
                len(audited),
                int(audited["qc_included"].sum()),
                int((~audited["qc_included"]).sum()),
                audited[["u", "v", "key"]].drop_duplicates().shape[0],
                audited["timestamp_label"].nunique(),
                int(audited["u"].eq(audited["v"]).sum()),
                int(audited.duplicated(["u", "v", "key", "timestamp_label"], keep=False).sum()),
                int(audited["status"].eq("ok").sum()),
            ],
        }
    )
    summary.to_csv(table_dir / "e00_qc_summary.csv", index=False)

    metrics = {
        "experiment_id": "E00",
        "seed": config["seed"],
        "source": str(source.relative_to(ROOT)),
        "configuration": qc,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
        **dict(zip(summary["metric"], summary["value"], strict=True)),
        "exclusion_reason_counts": dict(
            zip(reason_table["reason"], reason_table["observation_count"], strict=True)
        ),
        "integrity_checks_passed": int(integrity["passed"].sum()),
        "integrity_checks_total": len(integrity),
        "temporal_median_log_stress_range": float(
            temporal["median"].max() - temporal["median"].min()
        ),
        "road_class_total_variation_distance": float(
            representativeness["share_difference"].abs().sum() / 2
        ),
    }
    (metric_dir / "e00_qc_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audited.to_csv(processed_dir / "bandung_observed_v1.csv", index=False)
    save_distribution(
        audited["route_ratio"],
        figure_dir / "e00_route_ratio_distribution.svg",
        "Google distance / OSM edge length",
        "Bandung route-length consistency",
    )
    save_distribution(
        audited.loc[audited["qc_included"], "log_stress"],
        figure_dir / "e00_log_stress_distribution.svg",
        "log(Google time / OSM baseline time)",
        "Bandung included-observation stress",
    )
    save_temporal_variability(temporal, figure_dir / "e00_temporal_variability.svg")
    save_road_class_comparison(
        representativeness, figure_dir / "e00_road_class_representativeness.svg"
    )
    print(summary.to_string(index=False))
    print(f"Wrote E00 artifacts under {ROOT / 'results'}")


if __name__ == "__main__":
    main()
