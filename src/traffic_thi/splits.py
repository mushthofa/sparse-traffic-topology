"""Chronological partitions and leakage-aware fixed spatial observation protocols."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from traffic_thi.metr_la import sensor_ids, validate_timestamps


def temporal_partitions(
    index: pd.DatetimeIndex,
    rates: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> dict[str, slice]:
    """Floor train and validation counts; assign remaining rows to test, without shuffle."""
    validate_timestamps(index)
    if len(rates) != 3 or not np.isfinite(rates).all() or min(rates) <= 0:
        raise ValueError("Three positive finite partition rates are required")
    if not np.isclose(sum(rates), 1, rtol=0, atol=1e-12):
        raise ValueError("Partition rates must sum to one")
    train, validation = (int(np.floor(len(index) * r)) for r in rates[:2])
    if min(train, validation, len(index) - train - validation) < 1:
        raise ValueError("Every temporal partition must contain at least one timestamp")
    return {
        "train": slice(0, train),
        "validation": slice(train, train + validation),
        "test": slice(train + validation, len(index)),
    }


@dataclass(frozen=True)
class SensorMask:
    """Selection defined by IDs, independent of the input sensor-axis ordering."""

    repetition: int
    seed: int
    fraction: float
    selected_ids: tuple[str, ...]

    def observed(self, ids: object) -> np.ndarray:
        """Return a sensor-axis boolean mask in the requested, validated ID order."""
        ids = sensor_ids(ids)
        if not set(self.selected_ids) <= set(ids):
            raise ValueError("Selected sensors are absent from the supplied sensor axis")
        return np.isin(ids, self.selected_ids)


def nested_sensor_masks(
    ids: object,
    *,
    fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
) -> list[SensorMask]:
    """Rank IDs by SHA256(seed, ID); take nested floor(fraction*N) prefixes.

    Independent seeded hash rankings avoid dependence on NumPy versions or source
    column order. The JSON encoding unambiguously separates seed and sensor ID.
    """
    import json

    ids = sensor_ids(ids)
    if not fractions or any(not np.isfinite(f) or not 0 < f < 1 for f in fractions):
        raise ValueError("Observation fractions must be finite and strictly between zero and one")
    if tuple(sorted(set(fractions))) != tuple(fractions):
        raise ValueError("Observation fractions must be unique and increasing")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("Seeds must be distinct nonnegative integers")
    masks = []
    for repetition, seed in enumerate(seeds):
        ranked = sorted(
            ids,
            key=lambda key: (
                hashlib.sha256(
                    json.dumps([seed, key], ensure_ascii=True, separators=(",", ":")).encode()
                ).digest(),
                key,
            ),
        )
        for fraction in fractions:
            count = int(np.floor(fraction * len(ids)))
            if not 0 < count < len(ids):
                raise ValueError("Observation fraction must retain observed and withheld sensors")
            masks.append(SensorMask(repetition, seed, fraction, tuple(ranked[:count])))
    return masks


def protocol_masks(
    valid: np.ndarray,
    observed: np.ndarray,
    partitions: dict[str, slice],
    *,
    protocol: str = "A",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return permitted inputs O and held-out valid evaluation masks E by partition.

    A: observed sensors only at every time. B: all sensors during train, observed
    sensors only during validation/test. Targets outside E must not be scored.
    """
    valid = np.asarray(valid)
    observed = np.asarray(observed)
    if valid.dtype != bool or observed.dtype != bool or valid.ndim != 2:
        raise ValueError("valid and observed must be boolean time×sensor / sensor masks")
    if observed.shape != (valid.shape[1],):
        raise ValueError("Observation mask does not match sensor axis")
    if protocol not in {"A", "B"}:
        raise ValueError("protocol must be A (inductive) or B (transductive)")
    # Validate that the supplied partitions form exactly one chronological cover.
    if set(partitions) != {"train", "validation", "test"}:
        raise ValueError("Partitions must contain exactly train, validation, test")
    cursor = 0
    for name in ["train", "validation", "test"]:
        section = partitions[name]
        if section.start != cursor or section.step not in (None, 1) or section.stop <= cursor:
            raise ValueError("Partitions must be a contiguous nonempty chronological cover")
        cursor = section.stop
    if cursor != len(valid):
        raise ValueError("Partitions must cover exactly all timestamps")
    inputs = valid & observed[None, :]
    if protocol == "B":
        inputs[partitions["train"]] = valid[partitions["train"]]
    evaluation = {}
    for name, section in partitions.items():
        mask = np.zeros_like(valid)
        # B has no spatially held-out training targets (their history is available).
        if not (protocol == "B" and name == "train"):
            mask[section] = valid[section] & ~observed[None, :]
        evaluation[name] = mask
    return inputs, evaluation


def permitted_measurements(values: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """Materialize model inputs separately; unavailable entries are NaN, never zero."""
    values, inputs = np.asarray(values, dtype=float), np.asarray(inputs)
    if inputs.dtype != bool or values.shape != inputs.shape:
        raise ValueError("Inputs must be a boolean mask matching the measurement shape")
    if not np.isfinite(values[inputs]).all():
        raise ValueError("Permitted entries must be finite valid source readings")
    return np.where(inputs, values, np.nan)


def training_normalization(
    values: np.ndarray,
    inputs: np.ndarray,
    train: slice,
) -> dict[str, float | int]:
    """Fit one global mean/std using only permitted training entries (population std)."""
    allowed = permitted_measurements(values[train], inputs[train])
    sample = allowed[np.isfinite(allowed)]
    if not sample.size:
        raise ValueError("No valid permitted training readings for normalization")
    std = float(sample.std(ddof=0))
    return {
        "count": int(sample.size),
        "mean": float(sample.mean()),
        "standard_deviation_ddof0": std,
        "scale": std if std > 0 else 1.0,
    }
