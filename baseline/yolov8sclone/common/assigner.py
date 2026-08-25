"""TaskAlignedAssigner -- YOLOv8's dynamic label assignment.

This is the piece that most separates YOLOv8 from the anchor-based YOLOv5
recipe used by the CLAD-Net baseline. Instead of deciding which predictions
are positive from the *shape* of the ground-truth box (does it fit this
anchor within a factor of 4?), the assignment is decided from how well each
prediction is currently doing on both tasks at once:

    alignment = classification_score ** alpha  *  CIoU ** beta

For every ground-truth box, the `topk` best-aligned anchor points whose centre
falls inside the box become positives. An anchor claimed by several boxes goes
to the one it overlaps most. The alignment value is then rescaled per box and
used as the *soft* classification target, so a well-localised prediction is
asked for a higher class score than a poorly localised one.
"""

import torch
import torch.nn as nn

from .boxes import bbox_iou


def select_candidates_in_gts(xy_centers: torch.Tensor, gt_bboxes: torch.Tensor,
                             eps: float = 1e-9) -> torch.Tensor:
    """(b, n_boxes, n_anchors) mask: is this anchor centre inside this box?"""
    n_anchors = xy_centers.shape[0]
    batch_size, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
    deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2)
    return deltas.view(batch_size, n_boxes, n_anchors, -1).amin(3).gt_(eps)


def select_highest_overlaps(mask_pos: torch.Tensor, overlaps: torch.Tensor, n_max_boxes: int):
    """Give an anchor claimed by several boxes to the one it overlaps most."""
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:
        multi = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
        best = overlaps.argmax(1)
        is_best = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        is_best.scatter_(1, best.unsqueeze(1), 1)
        mask_pos = torch.where(multi, is_best, mask_pos).float()
        fg_mask = mask_pos.sum(-2)
    return mask_pos.argmax(-2), fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk: int = 10, num_classes: int = 2, alpha: float = 0.5,
                 beta: float = 6.0, eps: float = 1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        pd_scores  (b, n_anchors, nc)   sigmoid class scores
        pd_bboxes  (b, n_anchors, 4)    xyxy, image pixels
        anc_points (n_anchors, 2)       anchor centres, image pixels
        gt_labels  (b, n_max_boxes, 1)
        gt_bboxes  (b, n_max_boxes, 4)  xyxy, image pixels
        mask_gt    (b, n_max_boxes, 1)  which ground-truth slots are real

        Returns target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx.
        """
        batch_size = pd_scores.shape[0]
        n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if n_max_boxes == 0:
            return (torch.zeros_like(pd_scores[..., 0], dtype=torch.long),
                    torch.zeros_like(pd_bboxes),
                    torch.zeros_like(pd_scores),
                    torch.zeros_like(pd_scores[..., 0], dtype=torch.bool),
                    torch.zeros_like(pd_scores[..., 0], dtype=torch.long))

        mask_pos, align_metric, overlaps = self._positive_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, n_max_boxes)
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, n_max_boxes)
        target_labels, target_bboxes, target_scores = self._targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask, batch_size, n_max_boxes, device)

        # Rescale the alignment so that, per ground-truth box, the best-aligned
        # anchor is asked for a class score equal to its own IoU.
        align_metric = align_metric * mask_pos
        best_align = align_metric.amax(dim=-1, keepdim=True)
        best_overlap = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm = (align_metric * best_overlap / (best_align + self.eps)).amax(-2).unsqueeze(-1)
        return target_labels, target_bboxes, target_scores * norm, fg_mask.bool(), target_gt_idx

    def _positive_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                       mask_gt, n_max_boxes):
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self._box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt, n_max_boxes)
        mask_topk = self._topk_candidates(align_metric, mask_gt.expand(-1, -1, self.topk).bool())
        return mask_topk * mask_in_gts * mask_gt, align_metric, overlaps

    def _box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, n_max_boxes):
        batch_size, n_anchors = pd_scores.shape[0], pd_scores.shape[1]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([batch_size, n_max_boxes, n_anchors],
                               dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        scores = torch.zeros([batch_size, n_max_boxes, n_anchors],
                             dtype=pd_scores.dtype, device=pd_scores.device)

        index = torch.zeros([2, batch_size, n_max_boxes], dtype=torch.long)
        index[0] = torch.arange(batch_size).view(-1, 1).expand(-1, n_max_boxes)
        index[1] = gt_labels.squeeze(-1)
        scores[mask_gt] = pd_scores[index[0], :, index[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, n_anchors, -1)[mask_gt]
        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, ciou=True).squeeze(-1).clamp_(0)

        return scores.pow(self.alpha) * overlaps.pow(self.beta), overlaps

    def _topk_candidates(self, metrics: torch.Tensor, topk_mask: torch.Tensor) -> torch.Tensor:
        """Keep the `topk` highest-alignment anchors per ground-truth box."""
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        topk_idxs.masked_fill_(~topk_mask, 0)

        count = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            count.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        # An index that landed twice came from the masked_fill_ above, not from
        # two genuine top-k hits; drop it.
        count.masked_fill_(count > 1, 0)
        return count.to(metrics.dtype)

    def _targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask, batch_size,
                 n_max_boxes, device):
        batch_index = torch.arange(batch_size, dtype=torch.int64, device=device)[..., None]
        flat_index = target_gt_idx + batch_index * n_max_boxes

        target_labels = gt_labels.long().flatten()[flat_index].clamp_(0)
        target_bboxes = gt_bboxes.view(-1, 4)[flat_index]

        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1],
                                     self.num_classes), dtype=torch.int64, device=device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        keep = (fg_mask[:, :, None] > 0).repeat(1, 1, self.num_classes)
        return target_labels, target_bboxes, torch.where(keep, target_scores, 0).float()
