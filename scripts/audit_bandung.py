"""Generate the reproducible E00 Bandung data-audit artifacts."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from traffic_thi.quality import apply_observation_qc, exclusion_counts

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


def main() -> None:
    """Run E00 without modifying the immutable source snapshot."""
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    qc = config["bandung"]["qc"]
    source = ROOT / config["bandung"]["google_samples"]
    frame = pd.read_csv(source)
    audited = apply_observation_qc(
        frame,
        route_ratio_min=qc["route_ratio_soft_min"],
        route_ratio_max=qc["route_ratio_soft_max"],
        remove_self_loops=qc["remove_self_loops"],
    )

    metric_dir = ROOT / "results/metrics"
    table_dir = ROOT / "results/tables"
    figure_dir = ROOT / "results/figures"
    processed_dir = ROOT / "data/processed"
    for directory in (metric_dir, table_dir, figure_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    reason_table = exclusion_counts(audited)
    reason_table.to_csv(table_dir / "e00_exclusion_report.csv", index=False)
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
    print(summary.to_string(index=False))
    print(f"Wrote E00 artifacts under {ROOT / 'results'}")


if __name__ == "__main__":
    main()
