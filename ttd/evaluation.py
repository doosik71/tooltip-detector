"""Matching and metric helpers for tooltip evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class HungarianMatch:
    """A one-to-one assignment from one GT index to one prediction index."""

    gt_index: int
    prediction_index: int
    distance_px: float


def hungarian_match(
    gt_tips: list[tuple[float, float]],
    predictions: list[tuple[float, float, float]],
    max_distance_px: float,
) -> list[HungarianMatch]:
    """Return maximum-cardinality, minimum-distance assignments within a cap."""
    if max_distance_px <= 0:
        raise ValueError("max_distance_px must be positive")
    if not gt_tips or not predictions:
        return []

    gt = np.asarray(gt_tips, dtype=float)
    pred = np.asarray([(x, y) for x, y, _ in predictions], dtype=float)
    distances = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)
    n_gt, n_pred = distances.shape

    # Pad with dummy nodes. A valid match costs at most ``cap`` while leaving
    # both nodes unmatched costs ``2 * (cap + 1)``; invalid pairs are dearer
    # still. This gives maximum cardinality first, then minimum distance.
    unmatched_cost = max_distance_px + 1.0
    invalid_cost = 2.0 * unmatched_cost + max_distance_px
    cost = np.full((n_gt + n_pred, n_gt + n_pred), invalid_cost, dtype=float)
    cost[:n_gt, :n_pred] = np.where(
        distances <= max_distance_px, distances, invalid_cost
    )
    cost[:n_gt, n_pred:] = unmatched_cost
    cost[n_gt:, :n_pred] = unmatched_cost
    cost[n_gt:, n_pred:] = 0.0

    rows, cols = linear_sum_assignment(cost)
    return [
        HungarianMatch(int(row), int(col), float(distances[row, col]))
        for row, col in zip(rows, cols, strict=True)
        if row < n_gt and col < n_pred and distances[row, col] <= max_distance_px
    ]
