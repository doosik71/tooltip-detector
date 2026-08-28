"""The box arithmetic the detection metrics need.

Letterboxing and NMS are Ultralytics' job in this baseline -- `YOLO.predict()`
does both and hands back boxes in the original frame's pixel space -- so the
only thing left here is the IoU matrix `metrics.py` matches predictions with.
"""

import numpy as np


def box_iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -- (len(a), len(b))."""
    if len(a_xyxy) == 0 or len(b_xyxy) == 0:
        return np.zeros((len(a_xyxy), len(b_xyxy)), dtype=np.float32)
    lt = np.maximum(a_xyxy[:, None, :2], b_xyxy[None, :, :2])
    rb = np.minimum(a_xyxy[:, None, 2:], b_xyxy[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a_xyxy[:, 2] - a_xyxy[:, 0]) * (a_xyxy[:, 3] - a_xyxy[:, 1])
    area_b = (b_xyxy[:, 2] - b_xyxy[:, 0]) * (b_xyxy[:, 3] - b_xyxy[:, 1])
    return (inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)).astype(np.float32)
