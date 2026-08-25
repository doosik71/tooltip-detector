"""Label assignment and the paper's detection loss.

    Loss = 0.05 * Loss_Reg + 1.0 * Loss_Obj + 0.5 * Loss_Cls          (paper Eq. 3)

with Loss_Reg = 1 - CIoU (Eqs. 4-6) and Loss_Obj / Loss_Cls cross-entropy.

Those three weights are exactly YOLOv5's `box` / `obj` / `cls` gains, which --
together with the anchor + objectness head and CIoU regression -- is why the
assignment implemented here is YOLOv5's: match a ground-truth box to every
anchor whose width and height are within a factor of 4, on the cell it falls
in plus the two nearest neighbouring cells.

PAPER: the text says "cross-entropy" for Loss_Obj and Loss_Cls. Objectness is
a single logit and the classes are predicted independently, so both are
binary cross-entropy here -- the two-class form of the same loss.
"""

import torch
import torch.nn as nn

from .boxes import bbox_ciou

# Per-level objectness weights. The stride-8 level owns far more cells than
# stride-32, so without this its (mostly negative) objectness term dominates.
OBJ_BALANCE = (4.0, 1.0, 0.4)


class DetectionLoss:
    def __init__(self, model, box_gain: float = 0.05, obj_gain: float = 1.0,
                 cls_gain: float = 0.5, anchor_threshold: float = 4.0):
        self.num_classes = model.num_classes
        self.num_anchors = model.num_anchors
        self.anchors = model.anchors                      # (nl, na, 2), grid units
        self.num_levels = len(model.strides)
        self.box_gain, self.obj_gain, self.cls_gain = box_gain, obj_gain, cls_gain
        self.anchor_threshold = anchor_threshold
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def __call__(self, outputs, targets):
        """outputs: list of (B, na, ny, nx, 5+nc). targets: (nt, 6) normalised."""
        device = outputs[0].device
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)

        assignments = self._build_targets(outputs, targets.to(device))

        for level, prediction in enumerate(outputs):
            batch_index, anchor_index, grid_y, grid_x, target_box, target_cls, anchor = assignments[level]
            target_obj = torch.zeros(prediction.shape[:4], device=device, dtype=prediction.dtype)

            if len(batch_index):
                matched = prediction[batch_index, anchor_index, grid_y, grid_x]
                pxy = matched[:, 0:2].sigmoid() * 2.0 - 0.5
                pwh = (matched[:, 2:4].sigmoid() * 2.0) ** 2 * anchor
                iou = bbox_ciou(torch.cat([pxy, pwh], 1), target_box)
                loss_box = loss_box + (1.0 - iou).mean()

                target_obj[batch_index, anchor_index, grid_y, grid_x] = iou.detach().clamp(0).to(target_obj.dtype)

                if self.num_classes > 1:
                    one_hot = torch.zeros_like(matched[:, 5:])
                    one_hot[range(len(batch_index)), target_cls] = 1.0
                    loss_cls = loss_cls + self.bce(matched[:, 5:], one_hot)

            loss_obj = loss_obj + self.bce(prediction[..., 4], target_obj) * OBJ_BALANCE[level]

        batch_size = outputs[0].shape[0]
        loss_box = loss_box * self.box_gain
        loss_obj = loss_obj * self.obj_gain
        loss_cls = loss_cls * self.cls_gain
        total = (loss_box + loss_obj + loss_cls) * batch_size
        return total, {"box": loss_box.item(), "obj": loss_obj.item(), "cls": loss_cls.item()}

    def _build_targets(self, outputs, targets):
        """YOLOv5 assignment: anchor ratio filter + the two nearest neighbour cells."""
        device = targets.device
        num_targets = targets.shape[0]
        assignments = []

        anchor_ids = torch.arange(self.num_anchors, device=device, dtype=torch.float32)
        # (na, nt, 7): [image, class, cx, cy, w, h, anchor]
        expanded = torch.cat(
            (targets.repeat(self.num_anchors, 1, 1),
             anchor_ids[:, None, None].repeat(1, num_targets, 1)), 2)

        bias = 0.5
        neighbours = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
                                  device=device, dtype=torch.float32) * bias

        for level, prediction in enumerate(outputs):
            anchors = self.anchors[level].to(device)
            _, _, ny, nx, _ = prediction.shape
            gain = torch.tensor([1, 1, nx, ny, nx, ny, 1], device=device, dtype=torch.float32)

            scaled = expanded * gain
            if num_targets:
                ratio = scaled[..., 4:6] / anchors[:, None]
                keep = torch.max(ratio, 1.0 / ratio).max(2)[0] < self.anchor_threshold
                scaled = scaled[keep]

                grid_xy = scaled[:, 2:4]
                inverse_xy = torch.tensor([nx, ny], device=device, dtype=torch.float32) - grid_xy
                left, up = ((grid_xy % 1.0 < bias) & (grid_xy > 1.0)).T
                right, down = ((inverse_xy % 1.0 < bias) & (inverse_xy > 1.0)).T
                mask = torch.stack((torch.ones_like(left), left, up, right, down))
                scaled = scaled.repeat((5, 1, 1))[mask]
                offsets = (torch.zeros_like(grid_xy)[None] + neighbours[:, None])[mask]
            else:
                scaled = expanded[0][:0]
                offsets = torch.zeros((0, 2), device=device)

            batch_index = scaled[:, 0].long()
            target_cls = scaled[:, 1].long()
            grid_xy = scaled[:, 2:4]
            grid_wh = scaled[:, 4:6]
            cell = (grid_xy - offsets).long()
            grid_x = cell[:, 0].clamp_(0, nx - 1)
            grid_y = cell[:, 1].clamp_(0, ny - 1)
            anchor_index = scaled[:, 6].long()

            target_box = torch.cat((grid_xy - cell, grid_wh), 1)
            assignments.append((batch_index, anchor_index, grid_y, grid_x,
                                target_box, target_cls, anchors[anchor_index]))
        return assignments
