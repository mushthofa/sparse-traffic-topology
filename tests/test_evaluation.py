"""Metric examples are synthetic arithmetic, never reconstruction results."""

import numpy as np
import pandas as pd
import pytest

from traffic_thi.evaluation import (
    aggregate_repetitions,
    grouped_heldout_metrics,
    heldout_metrics,
)


def test_metrics_exclude_invalid_targets_and_unobserved_predictions() -> None:
    target = np.array([[10.0, np.nan, 0], [14, -1, np.inf]])
    predicted = np.array([[12.0, np.nan, np.inf], [10, -999, np.nan]])
    mask = np.ones(target.shape, dtype=bool)
    result = heldout_metrics(target, predicted, mask)
    assert result["valid_entries"] == 2
    assert result["mae"] == 3
    assert result["rmse"] == pytest.approx(np.sqrt(10))
    mask[0, 0] = False
    assert heldout_metrics(target, predicted, mask)["mae"] == 4
    predicted[1, 0] = np.nan
    with pytest.raises(ValueError, match="Predictions must be finite"):
        heldout_metrics(target, predicted, mask)


def test_empty_evaluation_and_zero_policy() -> None:
    assert heldout_metrics(np.array([0]), np.array([2]), np.array([True])) == {
        "valid_entries": 0,
        "mae": None,
        "rmse": None,
    }
    assert (
        heldout_metrics(np.array([0]), np.array([2]), np.array([True]), zero_policy="include")[
            "mae"
        ]
        == 2
    )
    with pytest.raises(ValueError, match="boolean"):
        heldout_metrics(np.array([1]), np.array([2]), np.array([1]))
    with pytest.raises(ValueError, match="identical shapes"):
        heldout_metrics(np.array([1]), np.array([2, 3]), np.array([True]))


def test_metrics_by_sensor_and_source_clock_hour() -> None:
    times = pd.DatetimeIndex(["2012-01-01 00:00", "2012-01-01 01:00"])
    target = np.array([[10.0, 20], [30, 40]])
    predicted = target + np.array([[1, 2], [3, 4]])
    mask = np.array([[False, True], [False, True]])
    sensors, hours = grouped_heldout_metrics(target, predicted, mask, times, ["a", "b"])
    assert sensors.iloc[0]["valid_entries"] == 0
    assert sensors.iloc[1]["mae"] == 3
    assert sensors.iloc[1]["rmse"] == pytest.approx(np.sqrt(10))
    assert hours.loc[hours.hour == 0, "mae"].iloc[0] == 2
    assert hours.loc[hours.hour == 1, "mae"].iloc[0] == 4
    assert hours.valid_entries.sum() == 2


def test_aggregation_keeps_macro_and_pooled_errors_distinct() -> None:
    records = [
        {
            "protocol": "A",
            "split": "test",
            "fraction": 0.2,
            "repetition": r,
            "seed": r + 42,
            "valid_entries": r + 1,
            "mae": float(r + 1),
            "rmse": float(r + 1),
        }
        for r in range(5)
    ]
    summary = aggregate_repetitions(records).iloc[0]
    assert summary["valid_entries_sum"] == 15
    assert summary["mae_mean"] == 3
    assert summary["mae_sample_sd"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert summary["pooled_mae"] == pytest.approx(55 / 15)
    assert summary["pooled_rmse"] == pytest.approx(np.sqrt(225 / 15))
    assert summary["rmse_mean"] != summary["pooled_rmse"]
    with pytest.raises(ValueError, match="complete mask repetitions"):
        aggregate_repetitions(records[:-1])
    records[1]["seed"] = records[0]["seed"]
    with pytest.raises(ValueError, match="distinct seeds"):
        aggregate_repetitions(records)


def test_all_empty_repetitions_do_not_report_zero_error() -> None:
    records = [
        {
            "protocol": "A",
            "split": "test",
            "fraction": 0.2,
            "repetition": r,
            "seed": r,
            "valid_entries": 0,
            "mae": None,
            "rmse": None,
        }
        for r in range(5)
    ]
    summary = aggregate_repetitions(records).iloc[0]
    assert summary["nonempty_repetitions"] == 0
    assert pd.isna(summary["mae_mean"])
    assert pd.isna(summary["pooled_rmse"])


def test_time_of_day_metrics_reject_unassigned_timestamp() -> None:
    values = np.array([[10.0], [20.0]])
    with pytest.raises(ValueError, match="Timestamps"):
        grouped_heldout_metrics(
            values,
            values,
            np.ones_like(values, dtype=bool),
            pd.DatetimeIndex(["2012-01-01", None]),
            ["a"],
        )
