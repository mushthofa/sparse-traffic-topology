from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from traffic_thi.quality import (
    QC_RATIO_PROFILES,
    add_log_stress,
    add_route_quality,
    summarize_edge_temporal_variability,
    summarize_qc_sensitivity,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/bandung/google_edge_samples.csv"
METRICS = ROOT / "results/metrics"
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures"


def save_route_ratio_figure(df: pd.DataFrame, path: Path) -> None:
    """Save a reviewable route-ratio histogram with all QC boundaries."""
    with plt.rc_context({"svg.hashsalt": "traffic-thi-e00"}):
        figure, axis = plt.subplots(figsize=(7.2, 4.2))
        axis.hist(df["route_ratio"].dropna(), bins=40, color="#4472c4", edgecolor="white")
        colors = {"very_strict": "#555555", "strict": "#d55e00", "moderate": "#009e73"}
        for name, (lower, upper) in QC_RATIO_PROFILES.items():
            axis.axvline(lower, color=colors[name], linestyle="--", linewidth=1.2, label=name)
            axis.axvline(upper, color=colors[name], linestyle="--", linewidth=1.2)
        axis.set(xlabel="Google distance / OSM edge length", ylabel="Observations")
        axis.legend(title="QC profile")
        figure.tight_layout()
        figure.savefig(path, format="svg", metadata={"Date": None})
        plt.close(figure)


def main() -> None:
    for directory in (METRICS, TABLES, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    df = add_route_quality(add_log_stress(df))
    qc = summarize_qc_sensitivity(df)
    variability = summarize_edge_temporal_variability(df)
    qc.to_csv(METRICS / "e00_qc_sensitivity.csv", index=False)
    variability.to_csv(TABLES / "e00_edge_temporal_variability.csv", index=False)
    save_route_ratio_figure(df, FIGURES / "e00_route_ratio_qc.svg")

    status_counts = df["status"].value_counts(dropna=False).to_string()
    quantiles = df["route_ratio"].quantile([0, 0.01, 0.05, 0.5, 0.95, 0.99, 1])
    report = "\n".join(
        [
            "E00 Bandung data audit",
            f"rows={len(df):,}",
            f"unique_edges={df['edge_id'].nunique():,}",
            f"time_states={df['timestamp_label'].nunique():,}",
            f"zero_variation_edges={int(variability['zero_variation'].sum()):,}",
            "status counts:",
            status_counts,
            "route_ratio quantiles:",
            quantiles.to_string(),
            "QC sensitivity (inclusive bounds):",
            qc.to_string(index=False),
        ]
    )
    (METRICS / "e00_audit_summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
