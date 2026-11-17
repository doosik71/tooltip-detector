#!/usr/bin/env python3
"""Train RTMDet-Pose from scratch on this repository's tool annotations.

One network learns, per frame, the tool boxes (RTMDet's detector) and the tip
of each tool (RTMPose's SimCC head on RoI features of the same network). See
common/model.py for the architecture and common/loss.py for the losses.

The training recipe is RTMDet's, scaled down to one GPU:
  * AdamW, weight decay 0.05 on conv/linear weights only
  * 1000-iteration linear warmup, constant lr for the first half, cosine
    decay to 0.05 x lr over the second half
  * ExpMomentumEMA (momentum 2e-4) over weights and BatchNorm statistics;
    validation and the saved model use the averaged weights
  * mosaic augmentation, switched off for the last `--stage2-epochs` epochs
No pretrained weights are used; everything starts from random initialisation.

Outputs, under baseline/mipsclone/data/model/<dataset>/:

    model.pt            best checkpoint by tip F1 @ 20 px on validation
    model-last.pt       last epoch, plus optimizer/EMA state for resuming
    train-status.json   epochs completed, best fitness, run arguments
    metric.csv          one row per epoch (losses, tool AP, tip metrics, lr, time)

Usage:
    ./baseline/mipsclone/run train-model --dataset cholec80 --device cuda:1
"""

import argparse
import copy
import csv
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

if True:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from common.dataset import ToolPoseDataset, available_datasets, collate, default_data_root
    from common.inference import (DEFAULT_CONF, DEFAULT_MAX_DET, DEFAULT_NMS_IOU, model_dir,
                                  save_checkpoint)
    from common.loss import RTMDetPoseLoss
    from common.metrics import DetectionEvaluator
    from common.model import (DEFAULT_BINS, DEFAULT_ROI_PADDING, DEFAULT_ROI_SIZE, SCALES, build,
                              parameter_count)
    from common.progress import progress
    from common.tipmetrics import TipEvaluator

WARMUP_ITERS = 1000
WARMUP_START = 1e-5
MIN_LR_FACTOR = 0.05
GRAD_CLIP_NORM = 10.0
FITNESS_CAP = "20px"


class ExpMomentumEMA:
    """mmdet's ExpMomentumEMA: the averaging momentum starts near 1 and decays
    exponentially to `momentum`, so the average tracks the raw weights early
    on and smooths them later. Buffers (BatchNorm statistics) are averaged too."""

    def __init__(self, model, momentum: float = 2e-4, gamma: float = 2000.0):
        self.ema = copy.deepcopy(model).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)
        self.momentum = momentum
        self.gamma = gamma
        self.steps = 0

    @torch.no_grad()
    def update(self, model):
        m = (1 - self.momentum) * math.exp(-float(1 + self.steps) / self.gamma) + self.momentum
        source = model.state_dict()
        for key, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                value.lerp_(source[key].detach(), m)
            else:
                value.copy_(source[key])
        self.steps += 1


def lr_factor(step: int, epoch_progress: float, epochs: int) -> float:
    """Warmup by iteration, then flat, then cosine over the second half."""
    warm = WARMUP_START + (1 - WARMUP_START) * min(1.0, step / WARMUP_ITERS)
    half = epochs / 2
    if epoch_progress < half:
        return warm
    t = min(1.0, (epoch_progress - half) / max(1e-9, epochs - half))
    return warm * (MIN_LR_FACTOR + (1 - MIN_LR_FACTOR) * 0.5 * (1 + math.cos(math.pi * t)))


def build_optimizer(model, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


@torch.no_grad()
def validate(model, loader, device, frame_scale: float) -> dict:
    """Tool AP over all detections, tip metrics over confident ones.

    Coordinates stay in letterboxed input pixels and are divided by the
    letterbox scale, so distances come out in original-frame pixels (the
    padding offset cancels in a distance)."""
    model.eval()
    detection_eval = DetectionEvaluator(1, ("tool",))
    tip_eval = TipEvaluator()
    for images, targets, _ in progress(loader, desc="  val", leave=False):
        images = images.to(device, non_blocking=True).float()
        results = model.predict(images, conf=0.001, max_det=DEFAULT_MAX_DET,
                                nms_iou=DEFAULT_NMS_IOU, pose_conf=DEFAULT_CONF)
        for i, result in enumerate(results):
            rows = targets[targets[:, 0] == i].numpy()
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            detections = np.concatenate([boxes, scores[:, None], np.zeros((len(boxes), 1))], 1)
            ground_truth = np.concatenate([np.zeros((len(rows), 1)), rows[:, 1:5]], 1)
            detection_eval.add(detections, ground_truth)

            confident = scores >= DEFAULT_CONF
            tips = result["keypoints"][:, 0].cpu().numpy()[confident] / frame_scale
            gt_tips = rows[rows[:, 7] > 0][:, 5:7] / frame_scale
            tip_eval.add([tuple(t) for t in gt_tips], [tuple(t) for t in tips])

    detection = detection_eval.compute()
    tip = tip_eval.compute()
    hungarian = tip["hungarian"][FITNESS_CAP]
    p, r = hungarian["precision"], hungarian["recall"]
    return {"tool_ap50": detection["map50"], "tool_ap50_95": detection["map50_95"],
            "tip_hit20": tip["hit_rate_20px_pct"], "tip_median_px": tip["median_dist_px"],
            "tip_precision20": p, "tip_recall20": r,
            "tip_f1_20": round(2 * p * r / (p + r), 4) if p + r > 0 else 0.0}


def append_metric_row(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_loader(dataset, batch_size, workers, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=collate, pin_memory=True, drop_last=shuffle,
                      persistent_workers=workers > 0)


def main():
    parser = argparse.ArgumentParser(description="Train RTMDet-Pose on a tooltip-detector dataset")
    parser.add_argument("--dataset", required=True, choices=available_datasets() or None)
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--scale", default="s", choices=tuple(SCALES),
                        help="RTMDet size (default: s, closest to the YOLOv8s baselines)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--stage2-epochs", type=int, default=10,
                        help="final epochs trained without mosaic (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--pose-weight", type=float, default=1.0,
                        help="weight of the SimCC keypoint loss against the detection loss")
    parser.add_argument("--roi-size", type=int, default=DEFAULT_ROI_SIZE)
    parser.add_argument("--roi-padding", type=float, default=DEFAULT_ROI_PADDING)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="use every Nth train frame; consecutive video frames are near-duplicates")
    parser.add_argument("--val-frames", type=int, default=2000,
                        help="cap on validation frames per epoch (0 = all)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore an existing model-last.pt and start over")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    args.output_dir = args.output_dir or model_dir(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "model.pt")
    last_path = os.path.join(args.output_dir, "model-last.pt")
    status_path = os.path.join(args.output_dir, "train-status.json")
    metric_path = os.path.join(args.output_dir, "metric.csv")

    arch = {"scale": args.scale, "num_keypoints": 1, "roi_size": args.roi_size,
            "roi_padding": args.roi_padding, "bins": args.bins}
    model = build(**arch).to(device)
    print(f"RTMDet-Pose-{args.scale}  {parameter_count(model) / 1e6:.3f} M parameters  [{device}]")

    train_set = ToolPoseDataset(args.dataset, "train", args.image_size, augment=True,
                                data_root=args.data_root, frame_stride=args.frame_stride)
    val_set = ToolPoseDataset(args.dataset, "val", args.image_size, augment=False,
                              data_root=args.data_root, limit=args.val_frames or None)
    frame = val_set.read_frame(0)[0]
    frame_scale = args.image_size / max(frame.shape[:2])
    print(f"train frames: {len(train_set):,}   val frames: {len(val_set):,}")
    val_loader = make_loader(val_set, args.batch_size, args.workers, shuffle=False)

    criterion = RTMDetPoseLoss(model, pose_weight=args.pose_weight)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = ExpMomentumEMA(model)

    start_epoch, step, best_fitness = 0, 0, -1.0
    if not args.no_resume and os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        ema.ema.load_state_dict(checkpoint["ema"])
        ema.steps = checkpoint["ema_steps"]
        start_epoch = checkpoint["epoch"] + 1
        step = checkpoint["step"]
        best_fitness = checkpoint.get("best_fitness", -1.0)
        print(f"resuming from epoch {start_epoch} (best tip F1@20 {best_fitness:.4f})")

    stage2_start = args.epochs - args.stage2_epochs
    train_loader, loader_mosaic = None, None
    for epoch in range(start_epoch, args.epochs):
        mosaic = epoch < stage2_start
        if train_loader is None or loader_mosaic != mosaic:
            # Workers hold their own copy of the dataset, so the switch needs
            # a new loader rather than a flag flipped in this process.
            train_set.mosaic = mosaic
            train_loader = make_loader(train_set, args.batch_size, args.workers, shuffle=True)
            loader_mosaic = mosaic
            if not mosaic:
                print("  mosaic off for the remaining epochs")

        model.train()
        epoch_start = time.perf_counter()
        running = {"cls": 0.0, "box": 0.0, "pose": 0.0}
        steps = 0
        bar = progress(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch_index, (images, targets, _) in enumerate(bar):
            factor = lr_factor(step, epoch + batch_index / len(train_loader), args.epochs)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor

            images = images.to(device, non_blocking=True).float()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
            # Losses (and the pose branch) run in fp32 outside autocast.
            loss, parts = criterion(*outputs, targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            step += 1

            for key in running:
                running[key] += parts[key]
            steps += 1
            bar.set_postfix(cls=f"{running['cls'] / steps:.3f}", box=f"{running['box'] / steps:.3f}",
                            pose=f"{running['pose'] / steps:.3f}")

        metrics = validate(ema.ema, val_loader, device, frame_scale)
        fitness = metrics["tip_f1_20"]
        elapsed = time.perf_counter() - epoch_start
        print(f"  epoch {epoch + 1}: tool AP50 {_fmt(metrics['tool_ap50'])}  "
              f"AP50:95 {_fmt(metrics['tool_ap50_95'])}  tip F1@20 {fitness:.4f}  "
              f"hit@20 {metrics['tip_hit20']:.2f} %  median {metrics['tip_median_px']} px  "
              f"({elapsed / 60:.1f} min)")

        append_metric_row(metric_path, {
            "epoch": epoch + 1,
            **{f"train_{k}_loss": round(v / max(1, steps), 5) for k, v in running.items()},
            **metrics,
            "lr": round(optimizer.param_groups[0]["lr"], 7),
            "mosaic": int(mosaic),
            "seconds": round(elapsed, 1),
        })

        if fitness > best_fitness:
            best_fitness = fitness
            save_checkpoint(best_path, ema.ema, args.image_size, args.dataset, epoch + 1, metrics)
            print(f"  saved {best_path} (tip F1@20 {fitness:.4f})")

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "ema": ema.ema.state_dict(),
                    "ema_steps": ema.steps, "epoch": epoch, "step": step,
                    "best_fitness": best_fitness, "arch": arch,
                    "image_size": args.image_size, "dataset": args.dataset}, last_path)

        with open(status_path, "w", encoding="utf-8") as handle:
            json.dump({"dataset": args.dataset, "epochs_completed": epoch + 1,
                       "epochs_total": args.epochs, "best_tip_f1_20": best_fitness,
                       "last_metrics": metrics, "args": vars(args)}, handle, indent=2)

    print(f"done. best tip F1@20 {best_fitness:.4f} -> {best_path}")


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
