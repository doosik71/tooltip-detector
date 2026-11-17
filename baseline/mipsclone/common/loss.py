"""Training losses of RTMDet-Pose.

    L = L_qfl + 2 · L_giou + pose_weight · L_simcc

Detection follows RTMDet: `DynamicSoftLabelAssigner` picks positives per GT
box, Quality Focal Loss trains the score towards the IoU of each positive's
predicted box, and GIoU trains the box, weighted by that same IoU. Both are
normalised by the sum of the positives' IoUs.

The pose term follows RTMPose's `KLDiscretLoss` (beta 10, label softmax) on
SimCC targets, with two changes of this project's own:

  * the KL divergence is summed over bins, not averaged. Averaging divides by
    the bin count (256 here) and, with the backbone now shared, would leave
    the pose gradient two orders of magnitude below the detection gradient;
  * the loss is averaged over instances rather than summed, so its size does
    not move with the number of tools in a batch.

The pose branch is trained on ground-truth tool boxes, each jittered in
position and size so that the branch sees boxes as imperfect as a detector's
before the detector is any good. A tip that falls outside its padded RoI
gets no keypoint loss (RTMPose's target weight 0).
"""

import torch
import torch.nn.functional as F

from .boxes import distance_to_box, giou_loss, pairwise_iou

_EPS = 1e-7


class DynamicSoftLabelAssigner:
    """RTMDet's assigner: cost = soft classification + IoU + soft centre prior,
    with a dynamic number of positives per GT (sum of its top-13 IoUs)."""

    def __init__(self, topk: int = 13, iou_weight: float = 3.0, soft_center_radius: float = 3.0):
        self.topk = topk
        self.iou_weight = iou_weight
        self.soft_center_radius = soft_center_radius

    @torch.no_grad()
    def __call__(self, priors, logits, boxes, gt_boxes, gt_labels):
        """priors (N, 3); logits (N, C); boxes (N, 4); gt_boxes (M, 4); gt_labels (M,).

        Returns labels (N,) with C for background, IoU targets (N,) and
        box targets (N, 4)."""
        n, c = logits.shape
        labels = torch.full((n,), c, dtype=torch.long, device=logits.device)
        metrics = logits.new_zeros(n)
        box_targets = boxes.new_zeros(n, 4)
        m = gt_boxes.shape[0]
        if m == 0:
            return labels, metrics, box_targets

        centres = priors[:, :2]
        inside = ((centres[:, None] > gt_boxes[None, :, :2]) &
                  (centres[:, None] < gt_boxes[None, :, 2:])).all(-1)       # (N, M)
        valid = inside.any(1)
        if not valid.any():
            return labels, metrics, box_targets

        v_boxes, v_logits, v_priors = boxes[valid], logits[valid], priors[valid]
        gt_centre = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        distance = (v_priors[:, None, :2] - gt_centre[None]).pow(2).sum(-1).sqrt() / v_priors[:, 2:3]
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        ious = pairwise_iou(v_boxes, gt_boxes)
        iou_cost = -torch.log(ious + _EPS) * self.iou_weight

        onehot = F.one_hot(gt_labels.long(), c).float()                     # (M, C)
        soft_label = onehot[None] * ious[..., None]                         # (V, M, C)
        scores = v_logits[:, None, :].expand(-1, m, -1)
        scale = (soft_label - scores.sigmoid()).abs().pow(2.0)
        cls_cost = (F.binary_cross_entropy_with_logits(scores, soft_label, reduction="none")
                    * scale).sum(-1)

        cost = cls_cost + iou_cost + soft_center_prior
        matched_iou, matched_gt, fg = self._dynamic_k(cost, ious)
        index = valid.nonzero().squeeze(1)[fg]
        labels[index] = gt_labels[matched_gt].long()
        metrics[index] = matched_iou
        box_targets[index] = gt_boxes[matched_gt]
        return labels, metrics, box_targets

    def _dynamic_k(self, cost, ious):
        v, m = cost.shape
        matching = torch.zeros_like(cost)
        top = ious.topk(min(self.topk, v), dim=0).values
        dynamic_k = top.sum(0).int().clamp(min=1)
        for g in range(m):
            index = cost[:, g].topk(int(dynamic_k[g]), largest=False).indices
            matching[index, g] = 1.0
        shared = matching.sum(1) > 1
        if shared.any():
            best = cost[shared].argmin(1)
            matching[shared] = 0.0
            matching[shared.nonzero().squeeze(1), best] = 1.0
        fg = matching.sum(1) > 0
        matched_gt = matching[fg].argmax(1)
        matched_iou = (matching * ious).sum(1)[fg]
        return matched_iou, matched_gt, fg


def quality_focal_loss(logits, labels, scores, beta: float = 2.0):
    """QFL summed over priors and classes. labels == C marks background."""
    sigmoid = logits.sigmoid()
    loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits),
                                              reduction="none") * sigmoid.pow(beta)
    pos = (labels < logits.shape[1]).nonzero().squeeze(1)
    if pos.numel():
        cls = labels[pos]
        p = logits[pos, cls]
        loss[pos, cls] = (F.binary_cross_entropy_with_logits(p, scores[pos], reduction="none")
                          * (scores[pos] - p.sigmoid()).abs().pow(beta))
    return loss.sum()


class RTMDetPoseLoss:
    def __init__(self, model, giou_weight: float = 2.0, pose_weight: float = 1.0,
                 simcc_sigma: float = 3.0, beta: float = 10.0, label_beta: float = 10.0,
                 roi_shift: float = 0.1, roi_scale: float = 0.1):
        if model.num_keypoints != 1:
            raise ValueError("the annotations carry one keypoint (the tip) per tool")
        self.model = model
        self.assigner = DynamicSoftLabelAssigner()
        self.giou_weight = giou_weight
        self.pose_weight = pose_weight
        self.simcc_sigma = simcc_sigma
        self.beta = beta
        self.label_beta = label_beta
        self.roi_shift = roi_shift
        self.roi_scale = roi_scale

    def __call__(self, feats, cls_scores, box_dists, targets):
        """targets: (n, 8) [batch, x1, y1, x2, y2, tip x, tip y, tip valid], input px.

        Returns (total loss, dict of float parts)."""
        model = self.model
        cls, dist = model.flatten(cls_scores, box_dists)
        cls, dist = cls.float(), dist.float()
        priors = model.priors(feats)
        boxes = distance_to_box(priors[None, :, :2], dist)
        targets = targets.to(cls.device)
        batch = cls.shape[0]

        all_labels, all_metrics, all_box_targets = [], [], []
        for i in range(batch):
            rows = targets[targets[:, 0] == i]
            gt_boxes = rows[:, 1:5]
            labels, metrics, box_targets = self.assigner(
                priors, cls[i].detach(), boxes[i].detach(), gt_boxes,
                torch.zeros(len(rows), dtype=torch.long, device=cls.device))
            all_labels.append(labels)
            all_metrics.append(metrics)
            all_box_targets.append(box_targets)
        labels = torch.cat(all_labels)
        metrics = torch.cat(all_metrics)
        box_targets = torch.cat(all_box_targets)
        flat_cls = cls.reshape(-1, cls.shape[-1])
        flat_boxes = boxes.reshape(-1, 4)

        norm = metrics.sum().clamp(min=1.0)
        loss_cls = quality_focal_loss(flat_cls, labels, metrics) / norm
        pos = labels < flat_cls.shape[1]
        if pos.any():
            loss_box = (giou_loss(flat_boxes[pos], box_targets[pos]) * metrics[pos]).sum() / norm
        else:
            loss_box = flat_boxes.sum() * 0.0

        loss_pose, n_pose = self._pose_loss(feats[0], targets)
        total = loss_cls + self.giou_weight * loss_box + self.pose_weight * loss_pose
        return total, {"cls": float(loss_cls), "box": float(loss_box),
                       "pose": float(loss_pose), "pose_n": n_pose}

    def _pose_loss(self, p3, targets):
        model = self.model
        if not len(targets):
            return p3.sum() * 0.0, 0
        boxes = targets[:, 1:5]
        size = boxes[:, 2:] - boxes[:, :2]
        centre = (boxes[:, :2] + boxes[:, 2:]) / 2
        # Detector-like errors: shift the centre and rescale each side.
        centre = centre + (torch.rand_like(centre) * 2 - 1) * self.roi_shift * size
        size = size * (1 + (torch.rand_like(size) * 2 - 1) * self.roi_scale)
        rois = model.rois(torch.cat([centre - size / 2, centre + size / 2], 1))

        bins = model.bins
        roi_size = (rois[:, 2:] - rois[:, :2]).clamp(min=1e-3)
        # Continuous bin coordinate of the tip; bin i is centred on i + 0.5.
        mu = (targets[:, 5:7] - rois[:, :2]) / roi_size * bins - 0.5
        inside = ((mu > -0.5) & (mu < bins - 0.5)).all(1)
        weight = targets[:, 7] * inside.float()
        n_valid = int((weight > 0).sum())
        if n_valid == 0:
            # Keep the branch in the graph so the backward pass stays uniform.
            sx, sy = model.pose(p3, targets[:1, 0], rois[:1])
            return (sx.sum() + sy.sum()) * 0.0, 0

        sx, sy = model.pose(p3, targets[:, 0], rois)                 # (R, K=1, bins)
        grid = torch.arange(bins, device=p3.device, dtype=torch.float32)
        loss = 0.0
        for axis, logits in enumerate((sx, sy)):
            gauss = torch.exp(-(grid[None] - mu[:, axis:axis + 1]) ** 2 / (2 * self.simcc_sigma ** 2))
            target = F.softmax(gauss * self.label_beta, -1)
            log_pred = F.log_softmax(logits[:, 0].float() * self.beta, -1)
            kl = (target * (torch.log(target.clamp(min=1e-12)) - log_pred)).sum(-1)
            loss = loss + kl
        return (loss * weight).sum() / weight.sum(), n_valid
