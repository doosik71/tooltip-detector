"""Box conversions, IoU variants, letterboxing and NMS."""

import cv2
import numpy as np
import torch
import torchvision


def xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    xy, wh = box[..., :2], box[..., 2:4] / 2
    return torch.cat([xy - wh, xy + wh], -1)


def xyxy_to_xywh(box: torch.Tensor) -> torch.Tensor:
    lt, rb = box[..., :2], box[..., 2:4]
    return torch.cat([(lt + rb) / 2, rb - lt], -1)


def bbox_ciou(pred_xywh: torch.Tensor, target_xywh: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete-IoU between matched box pairs -- the paper's Loss_Reg (Eqs. 4-6)."""
    p, t = pred_xywh, target_xywh
    p_x1, p_y1, p_x2, p_y2 = p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2
    t_x1, t_y1, t_x2, t_y2 = t[:, 0] - t[:, 2] / 2, t[:, 1] - t[:, 3] / 2, t[:, 0] + t[:, 2] / 2, t[:, 1] + t[:, 3] / 2

    inter = (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(0) * \
            (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(0)
    union = p[:, 2] * p[:, 3] + t[:, 2] * t[:, 3] - inter + eps
    iou = inter / union

    cw = torch.max(p_x2, t_x2) - torch.min(p_x1, t_x1)
    ch = torch.max(p_y2, t_y2) - torch.min(p_y1, t_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((t[:, 0] - p[:, 0]) ** 2 + (t[:, 1] - p[:, 1]) ** 2)

    v = (4 / np.pi ** 2) * (torch.atan(t[:, 2] / (t[:, 3] + eps)) -
                            torch.atan(p[:, 2] / (p[:, 3] + eps))).pow(2)
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


def non_max_suppression(prediction: torch.Tensor, conf_threshold: float = 0.25,
                        iou_threshold: float = 0.45, max_detections: int = 300
                        ) -> list[torch.Tensor]:
    """Class-wise NMS over decoded predictions.

    `prediction` is (B, N, 5 + nc) as returned by `common.model.decode`.
    Returns one (n, 6) tensor per image: [x1, y1, x2, y2, score, class].
    """
    results = []
    num_classes = prediction.shape[2] - 5
    for image_prediction in prediction:
        scores = image_prediction[:, 4:5] * image_prediction[:, 5:]
        best_score, best_class = scores.max(1)
        keep = best_score > conf_threshold
        if not keep.any():
            results.append(torch.zeros((0, 6), device=prediction.device))
            continue
        boxes = xywh_to_xyxy(image_prediction[keep, :4])
        best_score, best_class = best_score[keep], best_class[keep]
        order = torchvision.ops.batched_nms(boxes, best_score, best_class, iou_threshold)
        order = order[:max_detections]
        results.append(torch.cat([boxes[order], best_score[order, None],
                                  best_class[order, None].float()], 1))
    return results
