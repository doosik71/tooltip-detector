"""Box helpers: IoU / GIoU, distance decoding, letterboxing.

Boxes are xyxy in pixels everywhere in this package. The detector predicts,
per prior point, four distances (left, top, right, bottom) to the box edges,
which `distance_to_box` turns into xyxy -- RTMDet's `DistancePointBBoxCoder`.
"""

import cv2
import numpy as np
import torch


def distance_to_box(points: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """(…, 2) prior points + (…, 4) ltrb distances -> (…, 4) xyxy boxes."""
    x, y = points[..., 0], points[..., 1]
    return torch.stack([x - distances[..., 0], y - distances[..., 1],
                        x + distances[..., 2], y + distances[..., 3]], -1)


def pairwise_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """IoU between every box of `a` (n, 4) and every box of `b` (m, 4) -> (n, m)."""
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=eps)


def giou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - GIoU for aligned (n, 4) box pairs -> (n,). mmdet's `GIoULoss`."""
    lt = torch.maximum(pred[:, :2], target[:, :2])
    rb = torch.minimum(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_t = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = (area_p + area_t - inter).clamp(min=eps)
    iou = inter / union
    enclose_wh = (torch.maximum(pred[:, 2:], target[:, 2:]) -
                  torch.minimum(pred[:, :2], target[:, :2])).clamp(min=0)
    enclose = (enclose_wh[:, 0] * enclose_wh[:, 1]).clamp(min=eps)
    return 1 - (iou - (enclose - union) / enclose)


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


def letterbox(image: np.ndarray, size: int, pad_value: int = 114):
    """Resize preserving aspect ratio and pad to `size` x `size`.

    Returns (padded image, scale, pad_x, pad_y); `to_frame` undoes it.
    """
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), pad_value, dtype=image.dtype)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def to_frame(xy: np.ndarray, scale: float, pad_x: int, pad_y: int) -> np.ndarray:
    """Map (…, 2k) interleaved x, y coordinates from the letterboxed input back
    to the original frame. Works for xyxy boxes and for (x, y) points alike."""
    out = np.asarray(xy, dtype=np.float32).copy()
    out[..., 0::2] = (out[..., 0::2] - pad_x) / scale
    out[..., 1::2] = (out[..., 1::2] - pad_y) / scale
    return out
