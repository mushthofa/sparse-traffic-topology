"""Determinism, nestedness, ID alignment and leakage prevention contracts."""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from traffic_thi.metr_la import valid_speeds
from traffic_thi.splits import (
    nested_sensor_masks,
    permitted_measurements,
    protocol_masks,
    temporal_partitions,
    training_normalization,
)


def test_temporal_boundaries_are_chronological_and_exhaustive() -> None:
    index = pd.date_range("2012", periods=101, freq="5min")
    parts = temporal_partitions(index)
    assert parts == {"train": slice(0, 70), "validation": slice(70, 80), "test": slice(80, 101)}
    with pytest.raises(ValueError, match="at least one"):
        temporal_partitions(index[:5])
    for rates in [(0.7, 0.1, 0.3), (0.7, 0, 0.3), (np.nan, 0.1, 0.2)]:
        with pytest.raises(ValueError):
            temporal_partitions(index, rates)


def test_masks_are_nested_deterministic_and_sensor_order_independent() -> None:
    ids = [f"sensor_{i:03d}" for i in range(207)]
    masks = nested_sensor_masks(ids)
    assert masks == nested_sensor_masks(ids[::-1]) == nested_sensor_masks(ids)
    assert len(masks) == 20
    for repetition in range(5):
        group = masks[repetition * 4 : (repetition + 1) * 4]
        assert [len(mask.selected_ids) for mask in group] == [41, 82, 124, 165]
        for lower, upper in pairwise(group):
            assert set(lower.selected_ids) < set(upper.selected_ids)
        for mask in group:
            np.testing.assert_array_equal(mask.observed(ids), mask.observed(ids[::-1])[::-1])
    assert len({mask.selected_ids for mask in masks[::4]}) == 5
    # Adding or removing a time row cannot affect the spatial selection: no row input exists.


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fractions": (0, 0.4)},
        {"fractions": (0.4, 0.2)},
        {"fractions": (0.2, 0.2)},
        {"fractions": (float("nan"),)},
        {"seeds": (42, 42)},
        {"seeds": (-1,)},
    ],
)
def test_invalid_mask_specs_fail(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        nested_sensor_masks(["a", "b", "c", "d", "e"], **kwargs)


@pytest.mark.parametrize("protocol", ["A", "B"])
def test_input_and_evaluation_masks_and_normalization_prevent_leakage(protocol: str) -> None:
    ids = ["a", "b", "c", "d", "e"]
    values = np.arange(1, 101, dtype=float).reshape(20, 5)
    values[1, 1] = 0
    values[2, 2] = np.nan
    valid = valid_speeds(values)
    parts = temporal_partitions(pd.date_range("2012", periods=20, freq="5min"))
    for spatial in nested_sensor_masks(ids):
        observed = spatial.observed(ids)
        inputs, evaluation = protocol_masks(valid, observed, parts, protocol=protocol)
        permitted = permitted_measurements(values, inputs)
        assert not (inputs & evaluation["validation"]).any()
        assert not (inputs & evaluation["test"]).any()
        assert not np.isfinite(permitted[parts["validation"], ~observed]).any()
        assert not np.isfinite(permitted[parts["test"], ~observed]).any()
        assert not inputs[~valid].any()
        if protocol == "A":
            assert not np.isfinite(permitted[:, ~observed]).any()
            assert not (inputs & evaluation["train"]).any()
        else:
            np.testing.assert_array_equal(inputs[parts["train"]], valid[parts["train"]])
            assert not evaluation["train"].any()
        original = training_normalization(values, inputs, parts["train"])
        altered = values.copy()
        altered[parts["validation"]] = 1e9
        altered[parts["test"]] = -1e9
        if protocol == "A":
            altered[:, ~observed] = 9e9
        # This changes all forbidden targets; permitted training normalization is identical.
        altered_inputs, _ = protocol_masks(
            valid_speeds(altered), observed, parts, protocol=protocol
        )
        assert training_normalization(altered, altered_inputs, parts["train"]) == original
        expected = values[parts["train"]][inputs[parts["train"]]]
        assert original["count"] == expected.size
        assert original["mean"] == pytest.approx(expected.mean())
        for name, section in parts.items():
            if protocol == "B" and name == "train":
                continue
            assert evaluation[name][section].sum() == (valid[section] & ~observed).sum()


def test_no_valid_training_values_is_an_explicit_failure() -> None:
    values = np.full((10, 5), np.nan)
    with pytest.raises(ValueError, match="No valid permitted training"):
        training_normalization(values, np.zeros_like(values, dtype=bool), slice(0, 7))
    with pytest.raises(ValueError, match="finite"):
        permitted_measurements(values, np.ones_like(values, dtype=bool))
