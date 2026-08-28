"""YOLO26's detection loss: two assignments of the same loss, blended.

Each branch of the head is trained with the YOLOv8 detection loss:

    L_branch = 7.5 * L_box + 0.5 * L_cls + 1.5 * L_l1

There is no objectness term. The classification target is not 0/1 but the
alignment score produced by common.assigner.TaskAlignedAssigner, so a
prediction is asked for a high class score exactly to the extent that it also
localises well.

Two things differ from the YOLOv8 clone next door.

**No DFL.** `reg_max` is 1, so a box side is one regressed number rather than a
distribution over 16 bins. The third term is therefore an L1 on the four
distances, taken after normalising them by the image size so that a level's
stride does not decide how much that level contributes.

**Two branches, blended.** The one-to-many branch is assigned exactly as
YOLOv8's head is (topk=10). The one-to-one branch is assigned with topk=7 and
`topk2=1`, which leaves a single positive anchor per object and is what makes
NMS unnecessary at inference. The total is

    L = w_o2m * L_one2many + (1 - w_o2m) * L_one2one

with `w_o2m` decaying linearly from 0.8 to 0.1 over the run (`step_epoch()`).
Early on the model is mostly trained by the dense, easy-to-learn one-to-many
assignment; by the end it is mostly trained by the branch it will actually be
read from. The reported loss values are the one-to-one branch's, since that is
the branch inference uses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assigner import TaskAlignedAssigner
from .boxes import bbox_iou
from .model import bbox2dist, dist2bbox, make_anchors

BOX_GAIN = 7.5
CLS_GAIN = 0.5
L1_GAIN = 1.5

# Assignment settings of the two branches, and the blend schedule between them.
ONE2MANY_TOPK = 10
ONE2ONE_TOPK, ONE2ONE_TOPK2 = 7, 1
INITIAL_O2M_WEIGHT, FINAL_O2M_WEIGHT = 0.8, 0.1


class BboxLoss(nn.Module):
    """CIoU on the decoded box, L1 on the four raw distances."""

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, image_size, stride_tensor):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, ciou=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # Both sides are put back into pixels and then divided by the image
        # size, so a P5 anchor's error counts the same as a P3 anchor's. With
        # DFL the bin index did that normalisation implicitly.
        target_ltrb = self._normalise(bbox2dist(anchor_points, target_bboxes),
                                      stride_tensor, image_size)
        pred_ltrb = self._normalise(pred_dist, stride_tensor, image_size)
        loss_l1 = F.l1_loss(pred_ltrb[fg_mask], target_ltrb[fg_mask],
                            reduction="none").mean(-1, keepdim=True) * weight
        return loss_iou, loss_l1.sum() / target_scores_sum

    @staticmethod
    def _normalise(ltrb, stride_tensor, image_size):
        """Grid-unit distances -> fractions of the image's width and height."""
        pixels = ltrb * stride_tensor
        scale = torch.stack([image_size[1], image_size[0], image_size[1], image_size[0]])
        return pixels / scale


class BranchLoss:
    """The YOLOv8 detection loss over one head branch."""

    def __init__(self, model, topk: int, topk2: int | None = None,
                 gains: tuple[float, float, float] = (BOX_GAIN, CLS_GAIN, L1_GAIN)):
        self.num_classes = model.num_classes
        self.reg_max = model.reg_max
        self.strides = model.strides
        self.gains = gains
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=topk, num_classes=self.num_classes,
                                            alpha=0.5, beta=6.0, strides=self.strides,
                                            topk2=topk2)
        self.bbox_loss = BboxLoss()

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

    def __call__(self, preds: dict, targets: torch.Tensor):
        """preds: one branch's {boxes, scores, feats}. targets: (nt, 6) normalised."""
        features = preds["feats"]
        device = features[0].device
        batch_size = preds["boxes"].shape[0]

        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype

        image_size = torch.tensor(features[0].shape[2:], device=device, dtype=dtype) * self.strides[0]
        anchor_points, stride_tensor = make_anchors(features, self.strides)

        # ground truth
        prepared = self._preprocess(targets.to(device), batch_size,
                                    scale=image_size[[1, 0]])  # (w, h)
        gt_labels, gt_bboxes = prepared.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # No DFL integral to take: the head already predicts the four distances.
        pred_bboxes = dist2bbox(pred_distri, anchor_points, xywh=False)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)
        loss_cls = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        loss_box = torch.zeros(1, device=device)
        loss_l1 = torch.zeros(1, device=device)
        if fg_mask.sum():
            loss_box, loss_l1 = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                target_scores, target_scores_sum, fg_mask, image_size, stride_tensor)

        box_gain, cls_gain, l1_gain = self.gains
        loss_box = loss_box * box_gain
        loss_cls = loss_cls * cls_gain
        loss_l1 = loss_l1 * l1_gain
        total = (loss_box + loss_cls + loss_l1) * batch_size
        return total.squeeze(), {"box": float(loss_box), "cls": float(loss_cls),
                                 "l1": float(loss_l1)}


class DetectionLoss:
    """The end-to-end loss: both branches, blended on a decaying schedule."""

    def __init__(self, model, epochs: int,
                 gains: tuple[float, float, float] = (BOX_GAIN, CLS_GAIN, L1_GAIN)):
        self.one2many = BranchLoss(model, topk=ONE2MANY_TOPK, gains=gains)
        self.one2one = BranchLoss(model, topk=ONE2ONE_TOPK, topk2=ONE2ONE_TOPK2, gains=gains)
        self.epochs = epochs
        self.updates = 0
        self.o2m_weight = INITIAL_O2M_WEIGHT

    def step_epoch(self, completed_epochs: int | None = None) -> float:
        """Advance the blend schedule; call once per epoch, after training it.

        `completed_epochs` restores the schedule when a run is resumed, so a
        resumed epoch is weighted exactly as it would have been in one go.
        """
        self.updates = self.updates + 1 if completed_epochs is None else completed_epochs
        decay = max(1 - self.updates / max(self.epochs - 1, 1), 0.0)
        self.o2m_weight = decay * (INITIAL_O2M_WEIGHT - FINAL_O2M_WEIGHT) + FINAL_O2M_WEIGHT
        return self.o2m_weight

    def __call__(self, preds: dict, targets: torch.Tensor):
        """preds: the head's {one2many, one2one}. targets: (nt, 6) normalised."""
        loss_o2m, _ = self.one2many(preds["one2many"], targets)
        loss_o2o, parts = self.one2one(preds["one2one"], targets)
        o2o_weight = max(1.0 - self.o2m_weight, 0.0)
        # The reported parts are the one-to-one branch's: that is the branch
        # inference reads, so it is the one whose numbers mean something.
        return loss_o2m * self.o2m_weight + loss_o2o * o2o_weight, parts
