"""Tip-localisation metrics, comparable with the root project's eval-model.py.

Each detection of this model carries its tool's tips (one or two). Taken as a
set of points per frame, those tips are directly comparable with
tooltip-detector on the metrics the root project reports, provided the
matching rule is the same one. Both rules from `ttd/evaluation.py` and
`scripts/eval-model.py` are reproduced here:

  legacy     each GT tip takes its nearest predicted tip, with no
             exclusivity -- two GT tips may share one prediction. A frame with
             no predicted tip at all counts every GT tip in it as missed.
             Hit-rate @ N px is the share of GT tips whose nearest prediction
             is within N px.
  hungarian  a one-to-one assignment capped at N px, maximising the number of
             matches first and minimising total distance second, giving
             TP / FP / FN per cap.

Those two rules ignore which tool a tip belongs to. `PairEvaluator` adds the
measure this model exists for: does each tool come with its own tips? It is
this project's own version, for one or two tips per tool; the module is
otherwise copied from standalone/yolov8stooltip.

Distances are in the pixel space of the original frame (736 x 480).
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

    def add(self, gt_tips, predicted_tips) -> dict[float, set[int]]:
        """Score one frame; return, per cap, the indices of the GT tips the
        Hungarian assignment matched, so callers can record them per tip."""
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

        assigned = {}
        for cap in self.caps:
            matches = hungarian_match(gt, pred, cap)
            bucket = self._hungarian[cap]
            bucket["tp"] += len(matches)
            bucket["fn"] += len(gt) - len(matches)
            bucket["fp"] += len(pred) - len(matches)
            bucket["distances"].extend(distance for _, _, distance in matches)
            assigned[cap] = {gt_index for gt_index, _, _ in matches}
        return assigned

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


class PairEvaluator:
    """Does each ground-truth tool get a predicted tool *and its own* tips?

    Per frame, predicted tools are matched one-to-one to ground-truth tools by
    box IoU (Hungarian, maximising total IoU) and a match counts only at IoU
    >= `min_iou`. Each matched pair then compares the ground-truth tool's tips
    (one or two) with *that* prediction's tips (one or two), assigned to each
    other without regard to order. A tip close to some other tool's tip scores
    nothing here, which is the swap the tip-only rules above cannot see.

    pair_hit_rate_<N>px_pct       ground-truth tips whose own tool's matching
                                  predicted tip lies within N px, over all
                                  ground-truth tips (an unmatched tool or a
                                  missing second tip is a miss)
    two_tip_pair_hit_rate_<N>px_pct   the same over tools labelled with two tips
    tip2_precision / recall / f1  "this tool has a second tip", over matched
                                  pairs in frames whose label is trusted to
                                  mark a missing second tip (see dataset.py)
    """

    def __init__(self, caps: tuple[float, ...] = HIT_THRESHOLDS, min_iou: float = 0.5):
        self.caps = caps
        self.min_iou = min_iou
        self._n_tools = 0
        self._n_matched = 0
        self._n_tips = 0
        self._n_two_tips = 0
        self._distances: list[float] = []          # one per GT tip found, inf when missing
        self._two_tip_distances: list[float] = []
        self._tip2 = {"tp": 0, "fp": 0, "fn": 0}

    def add(self, gt_boxes, gt_tips, pred_boxes, pred_tips, trusted: bool = False) -> None:
        """Boxes (n, 4) xyxy; tips: one list of (x, y) points per tool."""
        gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
        pred_boxes = np.asarray(pred_boxes, float).reshape(-1, 4)
        self._n_tools += len(gt_boxes)
        for tips in gt_tips:
            self._n_tips += len(tips)
            if len(tips) == 2:
                self._n_two_tips += 2
        if not len(gt_boxes) or not len(pred_boxes):
            return
        lt = np.maximum(gt_boxes[:, None, :2], pred_boxes[None, :, :2])
        rb = np.minimum(gt_boxes[:, None, 2:], pred_boxes[None, :, 2:])
        inter = np.clip(rb - lt, 0, None).prod(2)
        area_gt = (gt_boxes[:, 2:] - gt_boxes[:, :2]).prod(1)
        area_pred = (pred_boxes[:, 2:] - pred_boxes[:, :2]).prod(1)
        iou = inter / (area_gt[:, None] + area_pred[None, :] - inter + 1e-9)
        for g, p in zip(*linear_sum_assignment(-iou)):
            if iou[g, p] < self.min_iou:
                continue
            self._n_matched += 1
            gt, pred = np.asarray(gt_tips[g], float), np.asarray(pred_tips[p], float)
            distances = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(distances)
            found = [float(distances[r, c]) for r, c in zip(rows, cols)]
            found += [float("inf")] * (len(gt) - len(found))
            self._distances.extend(found)
            if len(gt) == 2:
                self._two_tip_distances.extend(found)
            if trusted:
                has, said = len(gt) == 2, len(pred) == 2
                self._tip2["tp"] += has and said
                self._tip2["fp"] += said and not has
                self._tip2["fn"] += has and not said

    def compute(self) -> dict:
        distances = np.asarray(self._distances, dtype=float)
        two = np.asarray(self._two_tip_distances, dtype=float)
        finite = distances[np.isfinite(distances)]
        tp, fp, fn = self._tip2["tp"], self._tip2["fp"], self._tip2["fn"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        return {
            "n_gt_tools": self._n_tools,
            "n_gt_tips": self._n_tips,
            "n_gt_two_tip_tips": self._n_two_tips,
            "min_iou": self.min_iou,
            "box_matched_pct": round(self._n_matched / max(1, self._n_tools) * 100, 2),
            **{f"pair_hit_rate_{int(cap)}px_pct":
               round(int((distances <= cap).sum()) / max(1, self._n_tips) * 100, 2)
               for cap in self.caps},
            **{f"two_tip_pair_hit_rate_{int(cap)}px_pct":
               round(int((two <= cap).sum()) / max(1, self._n_two_tips) * 100, 2)
               for cap in self.caps},
            "pair_median_dist_px": _round(np.median(finite) if finite.size else None),
            "pair_mean_dist_px": _round(finite.mean() if finite.size else None),
            "tip2": {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
                     "recall": round(recall, 4),
                     "f1": round(2 * precision * recall / max(1e-9, precision + recall), 4)},
        }


def _round(value):
    return None if value is None else round(float(value), 2)
