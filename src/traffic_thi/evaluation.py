"""Algorithm-independent metrics on explicitly valid, spatially held-out targets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from traffic_thi.metr_la import sensor_ids, valid_speeds, validate_timestamps


def heldout_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    evaluation_mask: np.ndarray,
    *,
    zero_policy: str = "exclude",
) -> dict[str, float | int | None]:
    """Score E intersect valid(target); reject missing predictions on scored entries.

    Zero eligible entries yields count=0, MAE/RMSE=None, never a spurious zero error.
    Bad predictions cannot silently improve metrics by reducing coverage.
    """
    targets, predictions = np.asarray(targets, dtype=float), np.asarray(predictions, dtype=float)
    evaluation_mask = np.asarray(evaluation_mask)
    if targets.shape != predictions.shape or targets.shape != evaluation_mask.shape:
        raise ValueError("Targets, predictions and evaluation mask must have identical shapes")
    if evaluation_mask.dtype != bool:
        raise ValueError("Evaluation mask must have boolean dtype")
    mask = evaluation_mask & valid_speeds(targets, zero_policy=zero_policy)
    if not np.isfinite(predictions[mask]).all():
        raise ValueError("Predictions must be finite at every valid evaluated entry")
    count = int(mask.sum())
    residual = predictions[mask] - targets[mask]
    return {
        "valid_entries": count,
        "mae": float(np.abs(residual).mean()) if count else None,
        "rmse": float(np.sqrt(np.square(residual).mean())) if count else None,
    }


def grouped_heldout_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    evaluation_mask: np.ndarray,
    timestamps: pd.DatetimeIndex,
    ids: object,
    *,
    zero_policy: str = "exclude",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-sensor and hour-of-day metrics, preserving empty groups as null."""
    targets, predictions, evaluation_mask = map(np.asarray, (targets, predictions, evaluation_mask))
    ids = sensor_ids(ids)
    validate_timestamps(timestamps)
    if targets.shape != (len(timestamps), len(ids)):
        raise ValueError("Timestamp / sensor axes do not match target shape")
    heldout_metrics(targets, predictions, evaluation_mask, zero_policy=zero_policy)
    sensors = [
        {
            "sensor_id": key,
            **heldout_metrics(
                targets[:, i], predictions[:, i], evaluation_mask[:, i], zero_policy=zero_policy
            ),
        }
        for i, key in enumerate(ids)
    ]
    hours = []
    for hour in range(24):
        selected = timestamps.hour == hour
        hours.append(
            {
                "hour": hour,
                **heldout_metrics(
                    targets[selected],
                    predictions[selected],
                    evaluation_mask[selected],
                    zero_policy=zero_policy,
                ),
            }
        )
    return pd.DataFrame(sensors), pd.DataFrame(hours)


def aggregate_repetitions(
    records: Sequence[dict],
    *,
    expected_repetitions: int = 5,
) -> pd.DataFrame:
    """Macro mean/sample-SD and pooled errors, grouped by protocol, split, fraction.

    Input records contain repetition, seed, protocol, split, fraction and metrics.
    Require all distinct repetitions/seeds, including zero-coverage repetitions.
    No averaging of RMSE is mislabeled as pooled RMSE.
    """
    frame = pd.DataFrame(records)
    groups = ["protocol", "split", "fraction"]
    required = {*groups, "repetition", "seed", "valid_entries", "mae", "rmse"}
    if frame.empty or not required <= set(frame):
        raise ValueError("Complete repetition metric records are required")
    output = []
    for key, group in frame.groupby(groups, dropna=False, sort=True):
        if len(group) != expected_repetitions or group["repetition"].nunique() != len(group):
            raise ValueError("Expected distinct complete mask repetitions in each group")
        if group["seed"].nunique() != len(group):
            raise ValueError("Independent repetitions require distinct seeds")
        counts = group["valid_entries"].to_numpy(dtype=float)
        if (
            not np.isfinite(counts).all()
            or (counts < 0).any()
            or (counts != counts.astype(int)).any()
        ):
            raise ValueError("Valid-entry counts must be nonnegative integers")
        scored = group.loc[group["valid_entries"] > 0, ["mae", "rmse"]].astype(float)
        if not np.isfinite(scored.to_numpy()).all() or (scored < 0).any().any():
            raise ValueError("Nonempty repetitions need finite nonnegative metrics")
        total = int(counts.sum())
        output.append(
            {
                **dict(zip(groups, key, strict=True)),
                "repetitions": len(group),
                "nonempty_repetitions": len(scored),
                "valid_entries_sum": total,
                "mae_mean": scored["mae"].mean(),
                "mae_sample_sd": scored["mae"].std(),
                "rmse_mean": scored["rmse"].mean(),
                "rmse_sample_sd": scored["rmse"].std(),
                "pooled_mae": float((group["mae"].fillna(0) * counts).sum() / total)
                if total
                else None,
                "pooled_rmse": float(
                    np.sqrt((group["rmse"].fillna(0).pow(2) * counts).sum() / total)
                )
                if total
                else None,
            }
        )
    return pd.DataFrame(output)
