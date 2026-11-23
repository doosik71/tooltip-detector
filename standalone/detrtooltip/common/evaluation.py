"""One evaluation, shared by checkpoint selection (train-model) and eval-model.

Ground truth is read straight from the label files (pixels of the original
frame), so the numbers do not depend on the dataset's tensor layout, and any
model that yields annotation dicts can be scored the same way.

Predictions are annotation dicts with a `score` and optionally `tip2`. Tool AP
uses every prediction (the ranking needs low scores too); the tip and pair
metrics use only those scoring at least `conf`.

The checkpoint-selection score is the tip F1 @ 20 px: Hungarian one-to-one
matching of all predicted tips (`tip` and `tip2`) against all labelled tips.
"""

import numpy as np

from .metrics import DetectionEvaluator
from .model import CLASS_NAMES
from .tipmetrics import PairEvaluator, TipEvaluator

SELECTION_CAP = 20.0


def annotation_tips(item: dict) -> list[tuple[float, float]]:
    tips = [(item["tip"]["x"], item["tip"]["y"])]
    if "tip2" in item:
        tips.append((item["tip2"]["x"], item["tip2"]["y"]))
    return tips


def annotation_box(item: dict) -> tuple[float, float, float, float]:
    b = item["bbox"]
    return (b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"])


class Evaluation:
    def __init__(self, conf: float):
        self.conf = conf
        self.detection = DetectionEvaluator(len(CLASS_NAMES), CLASS_NAMES)
        self.tips = TipEvaluator()
        self.pairs = PairEvaluator()
        self.frames = 0

    def add(self, gt_tools: list[dict], predictions: list[dict], trusted: bool) -> None:
        """gt_tools as dataset.parse_label gives them; predictions as
        model.postprocess gives them, unfiltered."""
        self.frames += 1
        ground_truth = np.array([(0, *tool["box"]) for tool in gt_tools],
                                dtype=np.float32).reshape(-1, 5)
        detections = np.array([(*annotation_box(p), p["score"], 0) for p in predictions],
                              dtype=np.float32).reshape(-1, 6)
        self.detection.add(detections, ground_truth)

        kept = [p for p in predictions if p["score"] >= self.conf]
        gt_tips = [tool["tips"] for tool in gt_tools]
        pred_tips = [annotation_tips(p) for p in kept]
        self.tips.add([t for tips in gt_tips for t in tips], [t for tips in pred_tips for t in tips])
        self.pairs.add([tool["box"] for tool in gt_tools], gt_tips,
                       [annotation_box(p) for p in kept], pred_tips, trusted)

    def compute(self) -> dict:
        tips = self.tips.compute()
        selected = tips["hungarian"][f"{int(SELECTION_CAP)}px"]
        p, r = selected["precision"], selected["recall"]
        detection = self.detection.compute()
        return {
            "frames": self.frames,
            "conf": self.conf,
            "selection_tip_f1_20px": round(2 * p * r / max(1e-9, p + r), 4),
            "tool_ap50": detection["per_class"]["tool"]["ap50"],
            "tool_ap50_95": detection["per_class"]["tool"]["ap50_95"],
            "detection": detection,
            "tips": tips,
            "pairs": self.pairs.compute(),
        }
