from pathlib import Path
import pandas as pd

from traffic_thi.quality import add_log_stress, add_route_quality

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/bandung/google_edge_samples.csv"


def main() -> None:
    df = pd.read_csv(DATA)
    df = add_route_quality(add_log_stress(df))
    print(f"rows={len(df):,}")
    print(f"unique_edges={df['edge_id'].nunique():,}")
    print(f"time_states={df['timestamp_label'].nunique():,}")
    print("status counts:")
    print(df["status"].value_counts(dropna=False).to_string())
    print("route_ratio quantiles:")
    print(df["route_ratio"].quantile([0, .01, .05, .5, .95, .99, 1]).to_string())


if __name__ == "__main__":
    main()
