#!/usr/bin/env python3
"""Train yolov8stooltip: YOLOv8s with a pose head that outputs tool + tip pairs.

Each annotated tool is one target: its box and its tip (see common/dataset.py).
The head predicts both from the same anchor point, so the tip is learned as
part of the tool rather than as an unrelated `tip` class.

The recipe is the YOLOv8s clone baseline's, from scratch: 640 x 640, mosaic,
SGD with a short linear warmup and cosine decay, a weight EMA (`--no-ema` to
disable), and box/cls/dfl = 7.5/0.5/1.5 over TaskAlignedAssigner labels, plus
YOLOv8-pose's keypoint loss at 12.0.

The best checkpoint is picked by validation mAP@0.5:0.95 over `tool` and a
`tip` box drawn around each predicted tip (`--tip-box-size`), the same
fitness the baseline uses, so the two runs select their checkpoints alike.

Outputs, all under standalone/yolov8stooltip/data/model/<dataset>/ by default:

    model.pt            best checkpoint
    model-last.pt       last epoch, plus optimizer/EMA state for --resume
    train-status.json   epochs completed, best fitness, run arguments
    metric.csv          one row per epoch (losses, mAP, pair hits, lr, time)

Usage:
    ./standalone/yolov8stooltip/run train-model --dataset cholec80 --device cuda:1
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

    from common.boxes import non_max_suppression, xywh_to_xyxy
    from common.dataset import (DEFAULT_TIP_BOX_SIZE, SurgicalPoseDataset, available_datasets,
                                collate, default_data_root)
    from common.inference import model_dir, save_checkpoint
    from common.loss import DEFAULT_KPT_SIGMA, PoseLoss
    from common.metrics import DetectionEvaluator, ground_truth_with_tip_boxes, with_tip_boxes
    from common.model import YOLOv8Tooltip, build, decode, parameter_count
    from common.progress import progress
    from common.tipmetrics import PairEvaluator

WARMUP_EPOCHS = 3.0
FINAL_LR_FACTOR = 0.01      # cosine decays lr to lr * this
GRAD_CLIP_NORM = 10.0       # matches the reference YOLOv8 trainer
LOSS_PARTS = ("box", "cls", "dfl", "pose")
EVAL_NAMES = ("tool", "tip")


class ModelEMA:
    """Exponential moving average of the weights, evaluated instead of the raw
    model. Standard for this training recipe; disable with --no-ema."""

    def __init__(self, model, decay: float = 0.9999, warmup: int = 2000):
        self.ema = copy.deepcopy(model).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)
        self.updates = 0
        self.decay_fn = lambda n: decay * (1 - math.exp(-n / warmup))

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay_fn(self.updates)
        model_state = model.state_dict()
        for key, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(d).add_(model_state[key].detach(), alpha=1 - d)
            else:
                value.copy_(model_state[key])


def build_optimizer(model, lr: float, momentum: float, weight_decay: float):
    """Decay only convolution/linear weights -- not biases or BatchNorm."""
    decay, no_decay = [], []
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if name == "weight" and not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                decay.append(parameter)
            else:
                no_decay.append(parameter)
    return torch.optim.SGD([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, momentum=momentum, nesterov=True)


def lr_lambda(epoch: float, epochs: int) -> float:
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1e-9) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1e-9, epochs - WARMUP_EPOCHS)
    return FINAL_LR_FACTOR + (1 - FINAL_LR_FACTOR) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def as_float(outputs):
    """Head outputs in fp32 for the loss (see the AMP note in main)."""
    features, keypoints = outputs
    return [o.float() for o in features], keypoints.float()


@torch.no_grad()
def validate(model, loader, criterion, device, conf: float, iou: float,
             tip_box_size: float) -> dict:
    """Losses, tool/tip AP and pair hit-rates, in letterboxed input pixels."""
    model.eval()
    evaluator = DetectionEvaluator(len(EVAL_NAMES), EVAL_NAMES)
    pairs = PairEvaluator()
    totals = {key: 0.0 for key in LOSS_PARTS}
    batches = 0

    for images, targets, _ in progress(loader, desc="  val", leave=False):
        images = images.to(device, non_blocking=True).float().div_(255.0)
        outputs = as_float(model(images))
        _, parts = criterion(outputs, targets)
        for key in totals:
            totals[key] += parts[key]
        batches += 1

        detections = non_max_suppression(decode(outputs, model), model.num_classes, conf, iou)
        size = images.shape[-1]
        for i, detection in enumerate(detections):
            rows = targets[targets[:, 0] == i]
            tools = np.zeros((len(rows), 6), dtype=np.float32)
            if len(rows):
                tools[:, :4] = xywh_to_xyxy(rows[:, 2:6] * size).numpy()
                tools[:, 4:] = (rows[:, 6:8] * size).numpy()
            detection = detection.cpu().numpy()
            evaluator.add(with_tip_boxes(detection, tip_box_size),
                          ground_truth_with_tip_boxes(tools, tip_box_size))
            pairs.add(tools[:, :4], tools[:, 4:], detection[:, :4], detection[:, 6:8])

    metrics = evaluator.compute()
    metrics["pairs"] = pairs.compute()
    batches = max(1, batches)
    for key in LOSS_PARTS:
        metrics[f"val_{key}_loss"] = round(totals[key] / batches, 5)
    metrics["val_loss"] = round(sum(totals.values()) / batches, 5)
    return metrics


def append_metric_row(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Train yolov8stooltip (YOLOv8s + pose head) on a tooltip-detector dataset")
    parser.add_argument("--dataset", required=True, choices=available_datasets() or None,
                        help="dataset directory under data/dataset (e.g. cholec80)")
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--epochs", type=int, default=150, help="(default: 150)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="use every Nth train frame; consecutive video frames are near-duplicates")
    parser.add_argument("--val-frames", type=int, default=2000,
                        help="cap on validation frames per epoch (0 = all)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None,
                        help="override the checkpoint directory (default: data/model/<dataset>)")
    parser.add_argument("--scale", default="s", choices=tuple(YOLOv8Tooltip.SCALES),
                        help="YOLOv8 depth/width scale (default: s)")
    parser.add_argument("--kpt-sigma", type=float, default=DEFAULT_KPT_SIGMA,
                        help="OKS sigma of the tip in the keypoint loss; smaller is stricter "
                             f"(default: {DEFAULT_KPT_SIGMA:g})")
    parser.add_argument("--tip-box-size", type=float, default=DEFAULT_TIP_BOX_SIZE,
                        help="side of the box drawn around each tip for the validation tip AP "
                             f"only, in original-frame px (default: {DEFAULT_TIP_BOX_SIZE:g})")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="confidence threshold used when scoring validation mAP")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU for validation")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore an existing model-last.pt and start over")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    # resolved back into args so train-status.json records the real path
    args.output_dir = args.output_dir or model_dir(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "model.pt")
    last_path = os.path.join(args.output_dir, "model-last.pt")
    status_path = os.path.join(args.output_dir, "train-status.json")
    metric_path = os.path.join(args.output_dir, "metric.csv")

    arch = {"scale": args.scale}
    model = build(**arch).to(device)
    print(f"YOLOv8{args.scale} tooltip (pose)  {parameter_count(model) / 1e6:.3f} M parameters  "
          f"[{device}]  kpt sigma {args.kpt_sigma:g}")

    train_set = SurgicalPoseDataset(args.dataset, "train", args.image_size, augment=True,
                                    data_root=args.data_root, frame_stride=args.frame_stride)
    val_set = SurgicalPoseDataset(args.dataset, "val", args.image_size, augment=False,
                                  data_root=args.data_root, limit=args.val_frames or None)
    print(f"train frames: {len(train_set):,}   val frames: {len(val_set):,}")
    # Validation scores in letterboxed input pixels, so the tip box shrinks by
    # the letterbox scale, exactly as the baseline's tip labels do.
    frame_height, frame_width = val_set.read_frame(0)[0].shape[:2]
    val_tip_box = args.tip_box_size * args.image_size / max(frame_height, frame_width)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate, pin_memory=True)

    criterion = PoseLoss(model, kpt_sigma=args.kpt_sigma)
    optimizer = build_optimizer(model, args.lr, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: lr_lambda(e, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = None if args.no_ema else ModelEMA(model)

    start_epoch, best_fitness = 0, -1.0
    if not args.no_resume and os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if ema is not None and checkpoint.get("ema") is not None:
            ema.ema.load_state_dict(checkpoint["ema"])
            ema.updates = checkpoint.get("ema_updates", 0)
        start_epoch = checkpoint["epoch"] + 1
        best_fitness = checkpoint.get("best_fitness", -1.0)
        print(f"resuming from epoch {start_epoch} (best fitness {best_fitness:.4f})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.perf_counter()
        running = {key: 0.0 for key in LOSS_PARTS}
        steps = 0

        bar = progress(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for images, targets, _ in bar:
            images = images.to(device, non_blocking=True).float().div_(255.0)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
            # The loss runs in fp32 even under AMP. In half precision the CIoU
            # term's box-area products overflow and the whole loss goes NaN
            # within ~20 steps; the conv-heavy forward above keeps the speedup.
            loss, parts = criterion(as_float(outputs), targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            for key in running:
                running[key] += parts[key]
            steps += 1
            bar.set_postfix({key: f"{running[key] / steps:.4f}" for key in LOSS_PARTS})

        scheduler.step()
        evaluated = ema.ema if ema is not None else model
        metrics = validate(evaluated, val_loader, criterion, device, args.conf, args.iou,
                           val_tip_box)
        fitness = metrics["map50_95"] if metrics["map50_95"] is not None else -1.0
        elapsed = time.perf_counter() - epoch_start

        per_class, pairs = metrics["per_class"], metrics["pairs"]
        print(f"  epoch {epoch + 1}: val_loss {metrics['val_loss']:.4f}  "
              f"mAP@0.5:0.95 {_fmt(metrics['map50_95'])}  "
              f"tool AP50 {_fmt(per_class['tool']['ap50'])}  "
              f"tip AP50 {_fmt(per_class['tip']['ap50'])}  "
              f"pair@20 {pairs['pair_hit_rate_20px_pct']:.2f} %  "
              f"({elapsed / 60:.1f} min)")

        append_metric_row(metric_path, {
            "epoch": epoch + 1,
            **{f"train_{key}_loss": round(running[key] / max(1, steps), 5) for key in LOSS_PARTS},
            "val_loss": metrics["val_loss"],
            **{f"val_{key}_loss": metrics[f"val_{key}_loss"] for key in LOSS_PARTS},
            "map50": metrics["map50"], "map50_95": metrics["map50_95"],
            "tool_ap50": per_class["tool"]["ap50"], "tip_ap50": per_class["tip"]["ap50"],
            "pair_hit_10px_pct": pairs["pair_hit_rate_10px_pct"],
            "pair_hit_20px_pct": pairs["pair_hit_rate_20px_pct"],
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "seconds": round(elapsed, 1),
        })

        state = evaluated.state_dict()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "ema": ema.ema.state_dict() if ema is not None else None,
                    "ema_updates": ema.updates if ema is not None else 0,
                    "epoch": epoch, "best_fitness": best_fitness,
                    "arch": arch, "image_size": args.image_size,
                    "kpt_sigma": args.kpt_sigma, "dataset": args.dataset}, last_path)

        if fitness > best_fitness:
            best_fitness = fitness
            saved = build(**arch)
            saved.load_state_dict(state)
            save_checkpoint(best_path, saved, arch, args.image_size, args.kpt_sigma,
                            args.dataset, epoch + 1, metrics)
            print(f"  saved {best_path} (mAP@0.5:0.95 {fitness:.4f})")

        with open(status_path, "w", encoding="utf-8") as handle:
            json.dump({"dataset": args.dataset, "epochs_completed": epoch + 1,
                       "epochs_total": args.epochs, "best_map50_95": best_fitness,
                       "last_metrics": metrics, "args": vars(args)}, handle, indent=2)

    print(f"done. best mAP@0.5:0.95 {best_fitness:.4f} -> {best_path}")


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
