"""Box conversions, IoU variants and letterboxing.

There is no non-maximum suppression in this file, and that is the point:
YOLO26's head is end-to-end, so duplicate boxes are suppressed by the
one-to-one training objective rather than at inference. What replaces NMS is
`common.model.postprocess`, a top-k selection.
"""

import cv2
import numpy as np
import torch


def xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    xy, wh = box[..., :2], box[..., 2:4] / 2
    return torch.cat([xy - wh, xy + wh], -1)


def xyxy_to_xywh(box: torch.Tensor) -> torch.Tensor:
    lt, rb = box[..., :2], box[..., 2:4]
    return torch.cat([(lt + rb) / 2, rb - lt], -1)


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = False,
             ciou: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """IoU (optionally Complete-IoU) between broadcastable box tensors."""
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if not ciou:
        return iou

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    v = (4 / np.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


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

    Returns (padded image, scale, pad_x, pad_y) so predictions can be mapped
    back with `undo_letterbox`.
    """
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), pad_value, dtype=image.dtype)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def undo_letterbox(boxes_xyxy: np.ndarray, scale: float, pad_x: int, pad_y: int,
                   width: int, height: int) -> np.ndarray:
    """Map boxes from letterboxed input coordinates back to the original frame."""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    out = boxes_xyxy.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out
