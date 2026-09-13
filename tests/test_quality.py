import numpy as np
import pandas as pd
import pytest

from traffic_thi.quality import (
    add_log_stress,
    add_route_quality,
    apply_observation_qc,
    exclusion_counts,
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
