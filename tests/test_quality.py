
import numpy as np
import pandas as pd
import pytest

from traffic_thi.quality import (
    QC_RATIO_PROFILES,
    add_log_stress,
    add_route_quality,
    apply_observation_qc,
    exclusion_counts,
    qc_sensitivity,
    road_class_representativeness,
    summarize_edge_temporal_variability,
    summarize_qc_sensitivity,
    temporal_variability,
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


def test_invalid_values_have_missing_derived_features() -> None:
    df = pd.DataFrame(
        {
            "google_distance_m": [0.0],
            "osm_length_m": [100.0],
            "google_time_s": [0.0],
            "osm_time_s": [10.0],
        }
    )
    out = add_route_quality(add_log_stress(df))
    assert np.isnan(out.loc[0, "route_ratio"])
    assert np.isnan(out.loc[0, "route_confidence"])
    assert np.isnan(out.loc[0, "log_stress"])


def test_qc_retains_rows_and_reports_all_reasons() -> None:
    df = pd.DataFrame(
        {
            "edge_id": ["1_1_0", "2_3_0", "2_3_0"],
            "u": [1, 2, 2],
            "v": [1, 3, 3],
            "key": [0, 0, 0],
            "timestamp_label": ["mon_01", "mon_01", "mon_01"],
            "google_distance_m": [100.0, 200.0, 200.0],
            "osm_length_m": [100.0, 100.0, 100.0],
            "google_time_s": [20.0, 20.0, 20.0],
            "osm_time_s": [10.0, 10.0, 10.0],
            "status": ["ok", "error", "error"],
        }
    )
    out = apply_observation_qc(df)

    assert len(out) == len(df)
    assert list(out[["u", "v", "key"]].itertuples(index=False, name=None)) == [
        (1, 1, 0),
        (2, 3, 0),
        (2, 3, 0),
    ]
    assert out["value_provenance"].eq("observed_google").all()
    assert out.loc[0, "qc_exclusion_reasons"] == "self_loop"
    assert out.loc[1, "qc_exclusion_reasons"] == (
        "status_not_ok;duplicate_edge_time;route_ratio_above_max"
    )

    counts = exclusion_counts(out).set_index("reason")["observation_count"]
    assert counts["duplicate_edge_time"] == 2
    assert counts["self_loop"] == 1


def test_route_quality_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="sigma_q"):
        add_route_quality(pd.DataFrame(), sigma_q=0)


def test_qc_sensitivity_strict_is_nested_within_moderate() -> None:
    df = pd.DataFrame(
        {
            "u": [1, 2],
            "v": [2, 3],
            "key": [0, 0],
            "timestamp_label": ["mon_01", "mon_01"],
            "google_distance_m": [180.0, 110.0],
            "osm_length_m": [100.0, 100.0],
            "google_time_s": [20.0, 20.0],
            "osm_time_s": [10.0, 10.0],
            "status": ["ok", "ok"],
        }
    )
    profiles, summary = qc_sensitivity(
        df,
        {
            "strict": {
                "route_ratio_min": 0.7,
                "route_ratio_max": 1.5,
            },
            "moderate": {
                "route_ratio_min": 0.5,
                "route_ratio_max": 2.0,
            },
        },
    )

    assert profiles["strict"]["qc_included"].sum() == 1
    assert profiles["moderate"]["qc_included"].sum() == 2
    assert summary.set_index("profile").loc["strict", "excluded_observations"] == 1


def test_temporal_variability_only_uses_included_values() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_label": ["mon_01", "mon_01", "tue_01"],
            "day_name": ["mon", "mon", "tue"],
            "period": ["night", "night", "night"],
            "log_stress": [0.0, 100.0, 2.0],
            "qc_included": [True, False, True],
        }
    )
    report = temporal_variability(frame).set_index("timestamp_label")
    assert report.loc["mon_01", "mean"] == 0.0
    assert report.loc["tue_01", "mean"] == 2.0


def test_road_class_representativeness_preserves_directed_edges() -> None:
    population = pd.DataFrame(
        {
            "u": [1, 2, 2],
            "v": [2, 1, 3],
            "key": [0, 0, 0],
            "highway": ["a", "a", "b"],
        }
    )
    sampled = population.iloc[[0, 2]]
    report = road_class_representativeness(
        sampled, population
    ).set_index("highway")

    assert report["population_edge_count"].sum() == 3
    assert report["sample_edge_count"].sum() == 2
    assert report.loc["b", "sample_share"] == 0.5


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
    assert result.loc["a", "google_time_mean_s"] == pytest.approx(40 / 3)
    assert result.loc["a", "google_time_range_s"] == 10
    assert not bool(result.loc["a", "zero_variation"])
    assert bool(result.loc["b", "zero_variation"])
    assert (result.loc["b", ["u", "v", "key"]] == [2, 1, 1]).all()
