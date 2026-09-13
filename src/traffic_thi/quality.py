"""Quality-control helpers for observed traffic measurements."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

EDGE_KEY = ["u", "v", "key"]
OBSERVATION_KEY = [*EDGE_KEY, "timestamp_label"]


def add_route_quality(df: pd.DataFrame, sigma_q: float = 0.5) -> pd.DataFrame:
    """Add Google/OSM route-length ratio and a smooth confidence weight.

    Non-positive lengths do not define a route ratio and receive missing derived
    values rather than an infinite or misleading confidence value.
    """
    if sigma_q <= 0:
        raise ValueError("sigma_q must be positive")
    out = df.copy()
    valid = (out["google_distance_m"] > 0) & (out["osm_length_m"] > 0)
    out["route_ratio"] = np.nan
    out.loc[valid, "route_ratio"] = (
        out.loc[valid, "google_distance_m"] / out.loc[valid, "osm_length_m"]
    )
    out["route_confidence"] = np.nan
    out.loc[valid, "route_confidence"] = np.exp(
        -np.abs(np.log(out.loc[valid, "route_ratio"])) / sigma_q
    )
    return out


def add_log_stress(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``log_stress = log(google_time_s / osm_time_s)`` where defined."""
    out = df.copy()
    valid = (out["google_time_s"] > 0) & (out["osm_time_s"] > 0)
    out["log_stress"] = np.nan
    out.loc[valid, "log_stress"] = np.log(
        out.loc[valid, "google_time_s"] / out.loc[valid, "osm_time_s"]
    )
    return out


def _join_reasons(reason_masks: Iterable[tuple[str, pd.Series]], index: pd.Index) -> pd.Series:
    reasons = pd.Series("", index=index, dtype="string")
    for reason, mask in reason_masks:
        selected = mask.fillna(True)
        reasons.loc[selected] = reasons.loc[selected].map(
            lambda existing: f"{existing};{reason}" if existing else reason
        )
    return reasons


def apply_observation_qc(
    df: pd.DataFrame,
    *,
    route_ratio_min: float = 0.7,
    route_ratio_max: float = 1.5,
    remove_self_loops: bool = True,
) -> pd.DataFrame:
    """Derive audit fields and explicitly flag every excluded observation.

    All input rows are retained. ``qc_exclusion_reasons`` may contain multiple
    semicolon-delimited reasons, while ``qc_included`` is true only for rows with
    no reason. Directed edge columns ``(u, v, key)`` are never transformed.
    """
    if route_ratio_min <= 0 or route_ratio_max <= route_ratio_min:
        raise ValueError("route ratio bounds must satisfy 0 < min < max")

    out = add_route_quality(add_log_stress(df))
    duplicate = out.duplicated(OBSERVATION_KEY, keep=False)
    reasons: list[tuple[str, pd.Series]] = [
        ("status_not_ok", out["status"].ne("ok")),
        ("invalid_osm_length", out["osm_length_m"].le(0) | out["osm_length_m"].isna()),
        (
            "invalid_google_distance",
            out["google_distance_m"].le(0) | out["google_distance_m"].isna(),
        ),
        ("invalid_osm_time", out["osm_time_s"].le(0) | out["osm_time_s"].isna()),
        ("invalid_google_time", out["google_time_s"].le(0) | out["google_time_s"].isna()),
        ("duplicate_edge_time", duplicate),
        ("route_ratio_below_min", out["route_ratio"].lt(route_ratio_min)),
        ("route_ratio_above_max", out["route_ratio"].gt(route_ratio_max)),
    ]
    if remove_self_loops:
        reasons.append(("self_loop", out["u"].eq(out["v"])))

    out["qc_exclusion_reasons"] = _join_reasons(reasons, out.index)
    out["qc_included"] = out["qc_exclusion_reasons"].eq("")
    out["value_provenance"] = "observed_google"
    return out


def exclusion_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count individual QC reasons, including overlaps between reasons."""
    excluded = df.loc[~df["qc_included"], "qc_exclusion_reasons"]
    counts = excluded.str.split(";").explode().value_counts().sort_index()
    return counts.rename_axis("reason").reset_index(name="observation_count")


def qc_sensitivity(
    df: pd.DataFrame,
    profiles: dict[str, dict[str, float]],
    *,
    remove_self_loops: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Apply named QC profiles and summarize their inclusion sensitivity."""
    audited: dict[str, pd.DataFrame] = {}
    records: list[dict[str, object]] = []
    for name, bounds in profiles.items():
        result = apply_observation_qc(
            df,
            route_ratio_min=bounds["route_ratio_min"],
            route_ratio_max=bounds["route_ratio_max"],
            remove_self_loops=remove_self_loops,
        )
        result["qc_profile"] = name
        audited[name] = result
        counts = exclusion_counts(result).set_index("reason")["observation_count"].to_dict()
        records.append(
            {
                "profile": name,
                **bounds,
                "included_observations": int(result["qc_included"].sum()),
                "excluded_observations": int((~result["qc_included"]).sum()),
                **{f"excluded_{reason}": int(count) for reason, count in counts.items()},
            }
        )
    return audited, pd.DataFrame.from_records(records).fillna(0)


def temporal_variability(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize included log-stress variation for every observed time state."""
    included = df.loc[df["qc_included"] & df["log_stress"].notna()]
    return (
        included.groupby(["timestamp_label", "day_name", "period"], sort=False)["log_stress"]
        .agg(
            observation_count="size",
            mean="mean",
            median="median",
            standard_deviation="std",
            q05=lambda values: values.quantile(0.05),
            q95=lambda values: values.quantile(0.95),
        )
        .reset_index()
    )


def road_class_representativeness(
    sampled_edges: pd.DataFrame, population_edges: pd.DataFrame
) -> pd.DataFrame:
    """Compare sampled and full-network directed-edge highway-class shares."""
    sample = sampled_edges[list(EDGE_KEY) + ["highway"]].drop_duplicates(EDGE_KEY)
    population = population_edges[list(EDGE_KEY) + ["highway"]].drop_duplicates(EDGE_KEY)
    sample_counts = sample["highway"].fillna("missing").value_counts()
    population_counts = population["highway"].fillna("missing").value_counts()
    classes = population_counts.index.union(sample_counts.index).sort_values()
    report = pd.DataFrame(
        {
            "highway": classes,
            "population_edge_count": population_counts.reindex(classes, fill_value=0).to_numpy(),
            "sample_edge_count": sample_counts.reindex(classes, fill_value=0).to_numpy(),
        }
    )
    report["population_share"] = report["population_edge_count"] / len(population)
    report["sample_share"] = report["sample_edge_count"] / len(sample)
    report["share_difference"] = report["sample_share"] - report["population_share"]
    report["representation_ratio"] = report["sample_share"] / report["population_share"]
    return report
