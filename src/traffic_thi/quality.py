import numpy as np
import pandas as pd


QC_RATIO_PROFILES: dict[str, tuple[float, float]] = {
    "very_strict": (0.8, 1.25),
    "strict": (0.7, 1.5),
    "moderate": (0.5, 2.0),
}


def add_route_quality(df: pd.DataFrame, sigma_q: float = 0.5) -> pd.DataFrame:
    """Add Google/OSM route-length ratio and a smooth confidence weight."""
    out = df.copy()
    out["route_ratio"] = out["google_distance_m"] / out["osm_length_m"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["route_confidence"] = np.exp(-np.abs(np.log(out["route_ratio"])) / sigma_q)
    return out


def add_log_stress(df: pd.DataFrame) -> pd.DataFrame:
    """Add y=log(Google travel time / OSM baseline travel time)."""
    out = df.copy()
    valid = (out["google_time_s"] > 0) & (out["osm_time_s"] > 0)
    out["log_stress"] = np.nan
    out.loc[valid, "log_stress"] = np.log(
        out.loc[valid, "google_time_s"] / out.loc[valid, "osm_time_s"]
    )
    return out


def summarize_qc_sensitivity(
    df: pd.DataFrame,
    profiles: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Report route-ratio QC retention and exclusion counts for each profile.

    Rows with a non-``ok`` status, a missing/non-finite route ratio, or a ratio
    outside the inclusive profile bounds are excluded.  The mutually exclusive
    reason counts make every input observation auditable.
    """
    profiles = QC_RATIO_PROFILES if profiles is None else profiles
    quality = add_route_quality(df) if "route_ratio" not in df else df.copy()
    status_ok = quality["status"].eq("ok")
    finite_ratio = np.isfinite(quality["route_ratio"])

    rows: list[dict[str, int | float | str]] = []
    for name, (lower, upper) in profiles.items():
        within = quality["route_ratio"].between(lower, upper, inclusive="both")
        retained = status_ok & finite_ratio & within
        rows.append(
            {
                "profile": name,
                "ratio_min": lower,
                "ratio_max": upper,
                "total_rows": len(quality),
                "retained_rows": int(retained.sum()),
                "excluded_rows": int((~retained).sum()),
                "excluded_status": int((~status_ok).sum()),
                "excluded_missing_or_nonfinite_ratio": int((status_ok & ~finite_ratio).sum()),
                "excluded_ratio_below_min": int(
                    (status_ok & finite_ratio & quality["route_ratio"].lt(lower)).sum()
                ),
                "excluded_ratio_above_max": int(
                    (status_ok & finite_ratio & quality["route_ratio"].gt(upper)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_edge_temporal_variability(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize Google travel-time variability for each directed sampled road."""
    identity = ["edge_id", "u", "v", "key"]
    missing = [column for column in [*identity, "google_time_s"] if column not in df]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    grouped = df.groupby(identity, sort=True, dropna=False)["google_time_s"]
    result = grouped.agg(
        observation_count="size",
        unique_google_times="nunique",
        google_time_mean_s="mean",
        google_time_sd_s="std",
        google_time_min_s="min",
        google_time_max_s="max",
    ).reset_index()
    result["google_time_cv"] = result["google_time_sd_s"] / result["google_time_mean_s"]
    result["google_time_range_s"] = result["google_time_max_s"] - result["google_time_min_s"]
    result["zero_variation"] = result["unique_google_times"].le(1)
    return result
