import pandas as pd

from traffic_thi.quality import add_log_stress, add_route_quality


def test_quality_features() -> None:
    df = pd.DataFrame({
        "google_distance_m": [100.0],
        "osm_length_m": [100.0],
        "google_time_s": [20.0],
        "osm_time_s": [10.0],
    })
    out = add_route_quality(add_log_stress(df))
    assert out.loc[0, "route_ratio"] == 1.0
    assert out.loc[0, "route_confidence"] == 1.0
    assert out.loc[0, "log_stress"] > 0
