import numpy as np
import pandas as pd


def add_route_quality(df: pd.DataFrame, sigma_q: float = 0.5) -> pd.DataFrame:
    """Add Google/OSM route-length ratio and a smooth confidence weight."""
    out = df.copy()
    out["route_ratio"] = out["google_distance_m"] / out["osm_length_m"]
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
