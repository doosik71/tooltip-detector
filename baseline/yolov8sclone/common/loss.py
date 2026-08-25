"""YOLOv8's detection loss: BCE on classes, CIoU + DFL on boxes.

    Loss = 7.5 * L_box + 0.5 * L_cls + 1.5 * L_dfl

There is no objectness term. The classification target is not 0/1 but the
alignment score produced by common.assigner.TaskAlignedAssigner, so a
prediction is asked for a high class score exactly to the extent that it also
localises well.

L_dfl is the Distribution Focal Loss: each box side is predicted as a
distribution over `reg_max` integer bins, and the loss is the cross-entropy
towards the two bins bracketing the true distance, weighted by how close each
is. That is what lets the head regress sub-cell distances without anchors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assigner import TaskAlignedAssigner
from .boxes import bbox_iou
from .model import bbox2dist, dist2bbox, make_anchors

BOX_GAIN = 7.5
CLS_GAIN = 0.5
DFL_GAIN = 1.5


class BboxLoss(nn.Module):
    def __init__(self, reg_max: int):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, ciou=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # bbox2dist clamps to (max_distance - 0.01); the largest usable bin
        # index is reg_max - 1, so the distance must stay below that.
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
        loss_dfl = self._dfl(pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask])
        loss_dfl = (loss_dfl * weight).sum() / target_scores_sum
        return loss_iou, loss_dfl

    @staticmethod
    def _dfl(pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy towards the two bins bracketing the true distance."""
        lower = target.long()
        upper = lower + 1
        weight_lower = upper.to(target.dtype) - target
        weight_upper = 1 - weight_lower
        return (F.cross_entropy(pred_dist, lower.view(-1), reduction="none").view(lower.shape) * weight_lower +
                F.cross_entropy(pred_dist, upper.view(-1), reduction="none").view(lower.shape) * weight_upper
                ).mean(-1, keepdim=True)


class DetectionLoss:
    def __init__(self, model, box_gain: float = BOX_GAIN, cls_gain: float = CLS_GAIN,
                 dfl_gain: float = DFL_GAIN, topk: int = 10):
        self.num_classes = model.num_classes
        self.reg_max = model.reg_max
        self.num_outputs = model.num_outputs
        self.strides = model.strides
        self.dfl = model.detect.dfl
        self.gains = (box_gain, cls_gain, dfl_gain)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=topk, num_classes=self.num_classes,
                                            alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max)
        self.projection = torch.arange(self.reg_max, dtype=torch.float32)

    def _preprocess(self, targets: torch.Tensor, batch_size: int, scale: torch.Tensor):
        """(n, 6) [image, class, cx, cy, w, h] normalised -> padded per-image
        (b, n_max, 5) with [class, x1, y1, x2, y2] in input pixels."""
        device = scale.device
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=device)

        image_index = targets[:, 0]
        _, counts = image_index.unique(return_counts=True)
        out = torch.zeros(batch_size, int(counts.max()), 5, device=device)
        for i in range(batch_size):
            rows = targets[image_index == i]
            if rows.shape[0]:
                out[i, :rows.shape[0], 0] = rows[:, 1]
                cx, cy, w, h = (rows[:, 2] * scale[0], rows[:, 3] * scale[1],
                                rows[:, 4] * scale[0], rows[:, 5] * scale[1])
                out[i, :rows.shape[0], 1:] = torch.stack(
                    [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        return out

    def _decode_boxes(self, anchor_points, pred_dist):
        """DFL logits -> xyxy distances around each anchor point (grid units)."""
        b, a, c = pred_dist.shape
        projection = self.projection.to(pred_dist.device, pred_dist.dtype)
        expected = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(projection.type(pred_dist.dtype))
        return dist2bbox(expected, anchor_points, xywh=False)

    def __call__(self, outputs, targets):
        """outputs: list of (B, no, H, W). targets: (nt, 6) normalised."""
        device = outputs[0].device
        batch_size = outputs[0].shape[0]

        flattened = torch.cat([o.view(batch_size, self.num_outputs, -1) for o in outputs], 2)
        pred_distri, pred_scores = flattened.split((self.reg_max * 4, self.num_classes), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype

        image_size = torch.tensor(outputs[0].shape[2:], device=device, dtype=dtype) * self.strides[0]
        anchor_points, stride_tensor = make_anchors(outputs, self.strides)

        # ground truth
        prepared = self._preprocess(targets.to(device), batch_size,
                                    scale=image_size[[1, 0]])  # (w, h)
        gt_labels, gt_bboxes = prepared.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self._decode_boxes(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)
        loss_cls = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        loss_box = torch.zeros(1, device=device)
        loss_dfl = torch.zeros(1, device=device)
        if fg_mask.sum():
            target_bboxes = target_bboxes / stride_tensor
            loss_box, loss_dfl = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask)

        box_gain, cls_gain, dfl_gain = self.gains
        loss_box = loss_box * box_gain
        loss_cls = loss_cls * cls_gain
        loss_dfl = loss_dfl * dfl_gain
        total = (loss_box + loss_cls + loss_dfl) * batch_size
        return total.squeeze(), {"box": float(loss_box), "cls": float(loss_cls),
                                 "dfl": float(loss_dfl)}
