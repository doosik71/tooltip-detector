"""Losses: masked reconstruction (stage 1) and the set loss (stage 2).

Stage 2 targets, one dict per image, all coordinates normalised (0-1):

    boxes      (n, 4)    cx, cy, w, h
    tips       (n, 2, 2) up to two tips per tool, x then y
    tip_valid  (n, 2)    the tip exists and is inside the (augmented) image
    two_tips   (n,)      the label gives this tool two tips
    trusted    bool      the label source says where a second tip is *absent*
                         (manual, opus); False for labels from a one-tip model

Tips carry no order. For a two-tip tool the loss takes whichever of the two
slot assignments (A-t1, B-t2) or (A-t2, B-t1) costs less; a one-tip tool's
tip always goes to slot A. A tip cut off by augmentation is left out of the
sum rather than pulled towards the image edge.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .boxes import cxcywh_to_xyxy, generalized_box_iou

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
DEFAULT_WEIGHTS = {"cls": 2.0, "box": 5.0, "giou": 2.0, "tip": 5.0, "exist": 1.0}


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def random_patch_mask(batch: int, height: int, width: int, patch: int, ratio: float,
                      device) -> torch.Tensor:
    """(B, 1, H, W) with exactly round(ratio * patches) patches set to 1 per image."""
    rows, cols = height // patch, width // patch
    count = rows * cols
    hidden = round(ratio * count)
    order = torch.rand(batch, count, device=device).argsort(1)
    mask = torch.zeros(batch, count, device=device)
    mask.scatter_(1, order[:, :hidden], 1.0)
    mask = mask.view(batch, 1, rows, cols)
    return mask.repeat_interleave(patch, 2).repeat_interleave(patch, 3)


class ReconstructionLoss(nn.Module):
    """L1 on the hidden pixels against per-patch normalised targets (MAE's
    normalised pixel loss): each patch's channels are standardised by their
    own mean and variance, so the loss rewards structure rather than overall
    brightness."""

    def __init__(self, patch: int):
        super().__init__()
        self.patch = patch

    def target(self, normalised: torch.Tensor) -> torch.Tensor:
        b, c, h, w = normalised.shape
        p = self.patch
        x = normalised.view(b, c, h // p, p, w // p, p)
        mean = x.mean(dim=(3, 5), keepdim=True)
        var = x.var(dim=(3, 5), keepdim=True, unbiased=False)
        return ((x - mean) / (var + 1e-6).sqrt()).view(b, c, h, w)

    def forward(self, prediction: torch.Tensor, normalised: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        error = (prediction.float() - self.target(normalised.float())).abs() * mask
        return error.sum() / (mask.sum() * prediction.shape[1]).clamp(min=1.0)


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def tip_l1(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor,
           two: torch.Tensor) -> torch.Tensor:
    """Order-free L1 between tip slots and label tips; broadcasts.

    pred (..., 2, 2) slots A and B; gt (..., 2, 2); valid (..., 2) bool;
    two (...) bool -> (...)."""
    d = (pred[..., :, None, :] - gt[..., None, :, :]).abs().sum(-1)   # (..., slot, tip)
    v = valid.to(d.dtype)
    straight = d[..., 0, 0] * v[..., 0] + d[..., 1, 1] * v[..., 1]
    swapped = d[..., 1, 0] * v[..., 0] + d[..., 0, 1] * v[..., 1]
    return torch.where(two, torch.minimum(straight, swapped), d[..., 0, 0] * v[..., 0])


def focal_cost(logits: torch.Tensor) -> torch.Tensor:
    p = logits.sigmoid()
    neg = (1 - FOCAL_ALPHA) * p ** FOCAL_GAMMA * -(1 - p + 1e-8).log()
    pos = FOCAL_ALPHA * (1 - p) ** FOCAL_GAMMA * -(p + 1e-8).log()
    return pos - neg


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** FOCAL_GAMMA
    alpha_t = FOCAL_ALPHA * targets + (1 - FOCAL_ALPHA) * (1 - targets)
    return (alpha_t * loss).sum()


class SetCriterion(nn.Module):
    """Hungarian matching plus the weighted losses, on the final decoder
    layer (with the refined tips), every earlier layer and the two-stage
    proposals (class and box only)."""

    def __init__(self, weights: dict | None = None):
        super().__init__()
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    @torch.no_grad()
    def match(self, logits, boxes, tips, targets, with_tips: bool = True):
        """Per image, (query indices, target indices)."""
        w = self.weights
        indices = []
        for b, target in enumerate(targets):
            n = len(target["boxes"])
            if n == 0:
                empty = torch.zeros(0, dtype=torch.long)
                indices.append((empty, empty))
                continue
            cost = (w["cls"] * focal_cost(logits[b].float())[:, None]
                    + w["box"] * torch.cdist(boxes[b].float(), target["boxes"], p=1)
                    - w["giou"] * generalized_box_iou(cxcywh_to_xyxy(boxes[b].float()),
                                                      cxcywh_to_xyxy(target["boxes"])))
            if with_tips:
                cost = cost + w["tip"] * tip_l1(
                    tips[b].float()[:, None], target["tips"][None], target["tip_valid"][None],
                    target["two_tips"][None])
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
            indices.append((torch.as_tensor(rows, dtype=torch.long),
                            torch.as_tensor(cols, dtype=torch.long)))
        return indices

    def _layer_loss(self, out: dict, targets: list, num_boxes: float, with_tips: bool,
                    extra_tips: torch.Tensor | None = None) -> dict:
        logits, boxes = out["logits"].float(), out["boxes"].float()
        tips = out["tips"].float() if with_tips else None
        indices = self.match(logits, boxes, tips, targets, with_tips)
        device = logits.device

        batch_idx = torch.cat([torch.full_like(q, b) for b, (q, _) in enumerate(indices)]).to(device)
        query_idx = torch.cat([q for q, _ in indices]).to(device)
        target_cls = torch.zeros_like(logits)
        target_cls[batch_idx, query_idx] = 1.0
        losses = {"cls": sigmoid_focal_loss(logits, target_cls) / num_boxes}

        def gather(key):
            return torch.cat([t[key][g] for t, (_, g) in zip(targets, indices)]).to(device)

        matched_boxes = boxes[batch_idx, query_idx]
        gt_boxes = gather("boxes")
        if len(gt_boxes):
            losses["box"] = (matched_boxes - gt_boxes).abs().sum() / num_boxes
            giou = generalized_box_iou(cxcywh_to_xyxy(matched_boxes), cxcywh_to_xyxy(gt_boxes))
            losses["giou"] = (1 - giou.diagonal()).sum() / num_boxes
        else:
            losses["box"] = losses["giou"] = boxes.sum() * 0.0
        if not with_tips:
            return losses

        gt_tips, gt_valid, gt_two = gather("tips"), gather("tip_valid"), gather("two_tips")
        losses["tip"] = tip_l1(tips[batch_idx, query_idx], gt_tips, gt_valid, gt_two).sum() / num_boxes
        if extra_tips is not None:
            losses["tip_refined"] = tip_l1(extra_tips.float()[batch_idx, query_idx], gt_tips,
                                           gt_valid, gt_two).sum() / num_boxes

        trusted = torch.cat([torch.full((len(g),), bool(t["trusted"]), dtype=torch.bool)
                             for t, (_, g) in zip(targets, indices)]).to(device)
        exist_logits = out["exist_logits"].float()[batch_idx, query_idx]
        if trusted.any():
            losses["exist"] = F.binary_cross_entropy_with_logits(
                exist_logits[trusted], gt_two[trusted].float(), reduction="sum") / num_boxes
        else:
            losses["exist"] = exist_logits.sum() * 0.0
        return losses

    def forward(self, outputs: dict, targets: list) -> tuple[torch.Tensor, dict]:
        """-> (total loss, {part: float}) with parts summed over all layers."""
        num_boxes = max(1.0, float(sum(len(t["boxes"]) for t in targets)))
        w = self.weights
        w_of = {"cls": w["cls"], "box": w["box"], "giou": w["giou"], "tip": w["tip"],
                "tip_refined": w["tip"], "exist": w["exist"]}

        final = {"logits": outputs["logits"], "boxes": outputs["boxes"],
                 "tips": outputs["tips_coarse"], "exist_logits": outputs["exist_logits"]}
        groups = [self._layer_loss(final, targets, num_boxes, True, extra_tips=outputs["tips"])]
        groups += [self._layer_loss(aux, targets, num_boxes, True) for aux in outputs["aux"]]
        groups.append(self._layer_loss(outputs["enc"], targets, num_boxes, False))

        total = 0.0
        parts = {key: 0.0 for key in w_of}
        for group in groups:
            for key, value in group.items():
                total = total + w_of[key] * value
                parts[key] += float(value.detach())
        return total, parts
