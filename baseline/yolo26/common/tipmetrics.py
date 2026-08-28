"""Tip-localisation metrics, comparable with the root project's eval-model.py.

This baseline predicts a box for the `tip` class; its centre is the predicted tip
coordinate. That makes the detector directly comparable with tooltip-detector
on the metrics the root project reports, provided the matching rule is the
same one. Both rules from `ttd/evaluation.py` and `scripts/eval-model.py` are
reproduced here:

  legacy     each GT tip takes its nearest predicted tip, with no
             exclusivity -- two GT tips may share one prediction. A frame with
             no predicted tip at all counts every GT tip in it as missed.
             Hit-rate @ N px is the share of GT tips whose nearest prediction
             is within N px.
  hungarian  a one-to-one assignment capped at N px, maximising the number of
             matches first and minimising total distance second, giving
             TP / FP / FN per cap.

Distances are in the pixel space of the original frame (736 x 480), not of the
letterboxed network input.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

HIT_THRESHOLDS = (10.0, 20.0, 50.0)


def hungarian_match(gt_tips, predicted_tips, max_distance_px: float):
    """Maximum-cardinality, minimum-distance assignment within `max_distance_px`.

    Returns a list of (gt_index, prediction_index, distance). This mirrors
    `ttd.evaluation.hungarian_match` so the two projects' numbers mean the
    same thing.
    """
    if max_distance_px <= 0:
        raise ValueError("max_distance_px must be positive")
    if len(gt_tips) == 0 or len(predicted_tips) == 0:
        return []

    gt = np.asarray(gt_tips, dtype=float)
    pred = np.asarray(predicted_tips, dtype=float)
    distances = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)
    n_gt, n_pred = distances.shape

    unmatched_cost = max_distance_px + 1.0
    invalid_cost = 2.0 * unmatched_cost + max_distance_px
    cost = np.full((n_gt + n_pred, n_gt + n_pred), invalid_cost, dtype=float)
    cost[:n_gt, :n_pred] = np.where(distances <= max_distance_px, distances, invalid_cost)
    cost[:n_gt, n_pred:] = unmatched_cost
    cost[n_gt:, :n_pred] = unmatched_cost
    cost[n_gt:, n_pred:] = 0.0

    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c), float(distances[r, c])) for r, c in zip(rows, cols)
            if r < n_gt and c < n_pred and distances[r, c] <= max_distance_px]


class TipEvaluator:
    """Accumulates tip predictions frame by frame and reports both rules."""

    def __init__(self, caps: tuple[float, ...] = HIT_THRESHOLDS):
        self.caps = caps
        self._distances: list[float] = []
        self._n_gt = 0
        self._n_missed = 0
        self._hits = {cap: 0 for cap in caps}
        self._hungarian = {cap: {"tp": 0, "fp": 0, "fn": 0, "distances": []} for cap in caps}

    def add(self, gt_tips, predicted_tips) -> None:
        gt = list(gt_tips)
        pred = list(predicted_tips)
        self._n_gt += len(gt)

        if not pred:
            self._n_missed += len(gt)
        else:
            for gx, gy in gt:
                distance = min(float(np.hypot(gx - px, gy - py)) for px, py in pred)
                self._distances.append(distance)
                for cap in self.caps:
                    if distance <= cap:
                        self._hits[cap] += 1

        for cap in self.caps:
            matches = hungarian_match(gt, pred, cap)
            bucket = self._hungarian[cap]
            bucket["tp"] += len(matches)
            bucket["fn"] += len(gt) - len(matches)
            bucket["fp"] += len(pred) - len(matches)
            bucket["distances"].extend(distance for _, _, distance in matches)

    def compute(self) -> dict:
        distances = np.asarray(self._distances, dtype=float)
        safe = max(1, self._n_gt)
        result = {
            "n_gt_tips": self._n_gt,
            "n_missed": self._n_missed,
            "miss_rate_pct": round(self._n_missed / safe * 100, 2),
            "mean_dist_px": _round(distances.mean() if distances.size else None),
            "median_dist_px": _round(np.median(distances) if distances.size else None),
            "p90_dist_px": _round(np.percentile(distances, 90) if distances.size else None),
            **{f"hit_rate_{int(cap)}px_pct": round(self._hits[cap] / safe * 100, 2)
               for cap in self.caps},
            "hungarian": {},
        }
        for cap, bucket in self._hungarian.items():
            matched = np.asarray(bucket["distances"], dtype=float)
            tp, fp, fn = bucket["tp"], bucket["fp"], bucket["fn"]
            result["hungarian"][f"{int(cap)}px"] = {
                "max_distance_px": cap, "tp": tp, "fp": fp, "fn": fn,
                "precision": round(tp / max(1, tp + fp), 4),
                "recall": round(tp / max(1, tp + fn), 4),
                "mean_dist_px": _round(matched.mean() if matched.size else None),
                "median_dist_px": _round(np.median(matched) if matched.size else None),
            }
        return result


def _round(value):
    return None if value is None else round(float(value), 2)
