"""Detection metrics: AP@0.5, AP@0.5:0.95, precision and recall per class.

The accumulation follows the usual COCO/YOLO recipe: for each IoU threshold in
0.50:0.05:0.95, greedily match detections (highest score first) to unmatched
ground-truth boxes of the same class, then rank all detections by score and
integrate the precision-recall curve with 101-point interpolation.
"""

import numpy as np

from .boxes import box_iou_matrix

IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)


class DetectionEvaluator:
    """Accumulates per-image detections and ground truth, then reports AP."""

    def __init__(self, num_classes: int, class_names: tuple[str, ...] | None = None):
        self.num_classes = num_classes
        self.class_names = class_names or tuple(str(i) for i in range(num_classes))
        self._correct: list[np.ndarray] = []     # (n_det, n_iou) bool
        self._scores: list[np.ndarray] = []
        self._pred_classes: list[np.ndarray] = []
        self._gt_classes: list[np.ndarray] = []

    def add(self, detections: np.ndarray, ground_truth: np.ndarray) -> None:
        """detections: (n, 6) [x1,y1,x2,y2,score,class]; ground_truth: (m, 5) [class,x1,y1,x2,y2]."""
        self._gt_classes.append(ground_truth[:, 0].astype(int) if len(ground_truth)
                                else np.zeros(0, dtype=int))
        if not len(detections):
            return
        order = np.argsort(-detections[:, 4])
        detections = detections[order]
        self._scores.append(detections[:, 4])
        self._pred_classes.append(detections[:, 5].astype(int))

        correct = np.zeros((len(detections), len(IOU_THRESHOLDS)), dtype=bool)
        if len(ground_truth):
            iou = box_iou_matrix(detections[:, :4], ground_truth[:, 1:5])
            same_class = detections[:, 5][:, None] == ground_truth[:, 0][None, :]
            iou = np.where(same_class, iou, 0.0)
            for t, threshold in enumerate(IOU_THRESHOLDS):
                taken = np.zeros(len(ground_truth), dtype=bool)
                for d in range(len(detections)):
                    candidates = np.where((iou[d] >= threshold) & ~taken)[0]
                    if candidates.size:
                        best = candidates[np.argmax(iou[d, candidates])]
                        taken[best] = True
                        correct[d, t] = True
        self._correct.append(correct)

    def compute(self) -> dict:
        gt_counts = np.bincount(np.concatenate(self._gt_classes) if self._gt_classes
                                else np.zeros(0, dtype=int), minlength=self.num_classes)
        if not self._correct:
            return self._empty(gt_counts)

        correct = np.concatenate(self._correct)
        scores = np.concatenate(self._scores)
        pred_classes = np.concatenate(self._pred_classes)
        order = np.argsort(-scores)
        correct, scores, pred_classes = correct[order], scores[order], pred_classes[order]

        per_class = {}
        for c in range(self.num_classes):
            mask = pred_classes == c
            n_gt = int(gt_counts[c])
            if n_gt == 0 or not mask.any():
                per_class[self.class_names[c]] = {
                    "n_gt": n_gt, "n_pred": int(mask.sum()),
                    "ap50": None, "ap50_95": None, "precision": None, "recall": None}
                continue
            tp = correct[mask].cumsum(0)
            fp = (~correct[mask]).cumsum(0)
            recall = tp / n_gt
            precision = tp / np.maximum(tp + fp, 1e-9)
            aps = [_average_precision(recall[:, t], precision[:, t])
                   for t in range(len(IOU_THRESHOLDS))]
            per_class[self.class_names[c]] = {
                "n_gt": n_gt, "n_pred": int(mask.sum()),
                "ap50": round(float(aps[0]), 4),
                "ap50_95": round(float(np.mean(aps)), 4),
                # Precision and recall at the end of the ranking, i.e. over
                # every detection kept by the confidence threshold.
                "precision": round(float(precision[-1, 0]), 4),
                "recall": round(float(recall[-1, 0]), 4),
            }

        valid = [v for v in per_class.values() if v["ap50"] is not None]
        return {
            "per_class": per_class,
            "map50": round(float(np.mean([v["ap50"] for v in valid])), 4) if valid else None,
            "map50_95": round(float(np.mean([v["ap50_95"] for v in valid])), 4) if valid else None,
            "recall": round(float(np.mean([v["recall"] for v in valid])), 4) if valid else None,
            "precision": round(float(np.mean([v["precision"] for v in valid])), 4) if valid else None,
        }

    def _empty(self, gt_counts) -> dict:
        return {"per_class": {name: {"n_gt": int(gt_counts[i]), "n_pred": 0, "ap50": None,
                                     "ap50_95": None, "precision": None, "recall": None}
                              for i, name in enumerate(self.class_names)},
                "map50": None, "map50_95": None, "recall": None, "precision": None}


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP."""
    recall = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 0.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    precision = np.flip(np.maximum.accumulate(np.flip(precision)))
    points = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(points, recall, precision), points))
