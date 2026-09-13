import pandas as pd

from traffic_thi.quality import (
    QC_RATIO_PROFILES,
    add_log_stress,
    add_route_quality,
    summarize_edge_temporal_variability,
    summarize_qc_sensitivity,
)


def test_quality_features() -> None:
    df = pd.DataFrame(
        {
            "google_distance_m": [100.0],
            "osm_length_m": [100.0],
            "google_time_s": [20.0],
            "osm_time_s": [10.0],
        }
    )
    out = add_route_quality(add_log_stress(df))
    assert out.loc[0, "route_ratio"] == 1.0
    assert out.loc[0, "route_confidence"] == 1.0
    assert out.loc[0, "log_stress"] > 0


def test_qc_sensitivity_profiles_and_exclusion_accounting() -> None:
    df = pd.DataFrame(
        {
            "google_distance_m": [49.0, 50.0, 70.0, 150.0, 200.0, 201.0, 100.0],
            "osm_length_m": [100.0] * 7,
            "status": ["ok"] * 6 + ["error"],
        }
    )
    result = summarize_qc_sensitivity(df).set_index("profile")

    assert QC_RATIO_PROFILES["strict"] == (0.7, 1.5)
    assert QC_RATIO_PROFILES["moderate"] == (0.5, 2.0)
    assert result.loc["strict", "retained_rows"] == 2
    assert result.loc["moderate", "retained_rows"] == 4
    assert (result["retained_rows"] + result["excluded_rows"] == len(df)).all()
    assert (result["excluded_status"] == 1).all()


def test_edge_temporal_variability_preserves_directed_identity() -> None:
    df = pd.DataFrame(
        {
            "edge_id": ["a", "a", "a", "b", "b"],
            "u": [1, 1, 1, 2, 2],
            "v": [2, 2, 2, 1, 1],
            "key": [0, 0, 0, 1, 1],
            "google_time_s": [10.0, 10.0, 20.0, 5.0, 5.0],
        }
    )
    result = summarize_edge_temporal_variability(df).set_index("edge_id")

    assert result.loc["a", "unique_google_times"] == 2
    assert result.loc["a", "google_time_mean_s"] == 40 / 3
    assert result.loc["a", "google_time_range_s"] == 10
    assert not bool(result.loc["a", "zero_variation"])
    assert bool(result.loc["b", "zero_variation"])
    assert (result.loc["b", ["u", "v", "key"]] == [2, 1, 1]).all()
