#!/usr/bin/env python3
"""Train CLAD-Net on this repository's surgical-tool annotations.

Two classes are learned from one annotation file (see common/dataset.py):
`tool` (the annotated bounding box) and `tip` (a `--tip-box-size` px box centred on the
annotated tool tip), so one detector produces both the instrument box and the
tip coordinate the root project's metrics are defined on.

Training follows the paper's recipe -- 640 x 640, mosaic, SGD, lr 0.01, batch
16, loss 0.05/1.0/0.5 -- with three additions the paper does not mention but
that this loss/assignment recipe needs in practice, each switchable:
a short linear warmup, a cosine learning-rate decay, and a weight EMA
(`--no-ema` turns the last one off).

Outputs, all under baseline/cladnet/data/model/<dataset>/ by default:

    model.pt            best checkpoint by val mAP@0.5:0.95
    model-last.pt       last epoch, plus optimizer/EMA state for --resume
    train-status.json   epochs completed, best fitness, run arguments
    metric.csv          one row per epoch (losses, mAP, tip hit-rate, lr, time)

Usage:
    ./baseline/cladnet/run train-model --dataset cholec80
    ./baseline/cladnet/run train-model --dataset cholec80 --epochs 30 --device cuda:1
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
    from common.dataset import (DEFAULT_LABEL_SET, DEFAULT_TIP_BOX_SIZE, LABEL_SETS,
                                SurgicalDetectionDataset, available_datasets,
                                class_names, collate, default_data_root)
    from common.inference import model_dir, save_checkpoint
    from common.loss import DetectionLoss
    from common.metrics import DetectionEvaluator
    from common.progress import progress
    from common.model import build, decode, parameter_count

WARMUP_EPOCHS = 3.0
FINAL_LR_FACTOR = 0.01      # cosine decays lr to lr * this


class ModelEMA:
    """Exponential moving average of the weights, evaluated instead of the raw
    model. Not in the paper; standard for this training recipe and usually
    worth 1-2 points of mAP. Disable with --no-ema to train exactly as
    described."""

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


@torch.no_grad()
def validate(model, loader, criterion, device, conf: float, iou: float,
             names: tuple[str, ...]) -> dict:
    model.eval()
    evaluator = DetectionEvaluator(len(names), names)
    totals = {"box": 0.0, "obj": 0.0, "cls": 0.0}
    batches = 0

    for images, targets, _ in progress(loader, desc="  val", leave=False):
        images = images.to(device, non_blocking=True).float().div_(255.0)
        outputs = model(images)
        _, parts = criterion(outputs, targets)
        for key in totals:
            totals[key] += parts[key]
        batches += 1

        predictions = decode(outputs, model.anchors, model.strides)
        detections = non_max_suppression(predictions, conf, iou)
        size = images.shape[-1]
        for i, detection in enumerate(detections):
            rows = targets[targets[:, 0] == i]
            ground_truth = np.zeros((len(rows), 5), dtype=np.float32)
            if len(rows):
                ground_truth[:, 0] = rows[:, 1].numpy()
                ground_truth[:, 1:] = xywh_to_xyxy(rows[:, 2:] * size).numpy()
            evaluator.add(detection.cpu().numpy(), ground_truth)

    metrics = evaluator.compute()
    batches = max(1, batches)
    metrics["val_box_loss"] = round(totals["box"] / batches, 5)
    metrics["val_obj_loss"] = round(totals["obj"] / batches, 5)
    metrics["val_cls_loss"] = round(totals["cls"] / batches, 5)
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
    parser = argparse.ArgumentParser(description="Train CLAD-Net on a tooltip-detector dataset")
    parser.add_argument("--dataset", required=True, choices=available_datasets() or None,
                        help="dataset directory under data/dataset (e.g. cholec80)")
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--epochs", type=int, default=150,
                        help="matches the paper's schedule (default: 150)")
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
                        help="override the checkpoint directory; normally left alone, "
                             "since --label-set already picks "
                             "data/model/<dataset>/<label-set>")
    parser.add_argument("--neck-channels", type=int, default=112)
    parser.add_argument("--head-hidden", type=int, default=96)
    parser.add_argument("--tip-box-size", type=float, default=DEFAULT_TIP_BOX_SIZE,
                        help="side of the square box drawn around each annotated tip, in "
                             f"original-frame px (default: {DEFAULT_TIP_BOX_SIZE:g})")
    parser.add_argument("--label-set", default=DEFAULT_LABEL_SET, choices=tuple(LABEL_SETS),
                        help="which classes to learn: `tooltip` for the tool box and the "
                             "tip box, `tiponly` for the tip box alone. Also names the "
                             f"output directory (default: {DEFAULT_LABEL_SET})")
    parser.add_argument("--rm-combine", default="sum", choices=("sum", "mean"),
                        help="RM weight combination; see common.modules.RM")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="confidence threshold used when scoring validation mAP")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU for validation")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore an existing model-last.pt and start over")
    args = parser.parse_args()

    if args.batch_size < 2:
        # MSAB/RM pool to 1 x 1 before a BatchNorm, which needs >= 2 samples.
        parser.error("--batch-size must be at least 2")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    # resolved back into args so train-status.json records the real path
    args.output_dir = args.output_dir or model_dir(args.dataset, args.label_set)
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "model.pt")
    last_path = os.path.join(args.output_dir, "model-last.pt")
    status_path = os.path.join(args.output_dir, "train-status.json")
    metric_path = os.path.join(args.output_dir, "metric.csv")

    arch = {"neck_channels": args.neck_channels, "head_hidden": args.head_hidden,
            "rm_combine": args.rm_combine}
    names = class_names(args.label_set)
    model = build(num_classes=len(names), **arch).to(device)
    print(f"CLAD-Net  {parameter_count(model) / 1e6:.3f} M parameters  (paper: 7.5 M)  "
          f"[{device}]  tip box {args.tip_box_size:g} px")

    train_set = SurgicalDetectionDataset(args.dataset, "train", args.image_size, augment=True,
                                         data_root=args.data_root, frame_stride=args.frame_stride,
                                         tip_box_size=args.tip_box_size,
                                         label_set=args.label_set)
    val_set = SurgicalDetectionDataset(args.dataset, "val", args.image_size, augment=False,
                                       data_root=args.data_root,
                                       limit=args.val_frames or None,
                                       tip_box_size=args.tip_box_size,
                                       label_set=args.label_set)
    print(f"train frames: {len(train_set):,}   val frames: {len(val_set):,}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate, pin_memory=True)

    criterion = DetectionLoss(model)
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
        running = {"box": 0.0, "obj": 0.0, "cls": 0.0}
        steps = 0

        bar = progress(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for images, targets, _ in bar:
            images = images.to(device, non_blocking=True).float().div_(255.0)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, parts = criterion(model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            for key in running:
                running[key] += parts[key]
            steps += 1
            bar.set_postfix(box=f"{running['box'] / steps:.4f}",
                            obj=f"{running['obj'] / steps:.4f}",
                            cls=f"{running['cls'] / steps:.4f}")

        scheduler.step()
        evaluated = ema.ema if ema is not None else model
        metrics = validate(evaluated, val_loader, criterion, device, args.conf,
                           args.iou, names)
        fitness = metrics["map50_95"] if metrics["map50_95"] is not None else -1.0
        elapsed = time.perf_counter() - epoch_start

        per_class = metrics["per_class"]
        print(f"  epoch {epoch + 1}: val_loss {metrics['val_loss']:.4f}  "
              f"mAP@0.5 {_fmt(metrics['map50'])}  mAP@0.5:0.95 {_fmt(metrics['map50_95'])}  "
              f"tool AP50 {_fmt(_ap50(per_class, 'tool'))}  "
              f"tip AP50 {_fmt(_ap50(per_class, 'tip'))}  "
              f"({elapsed / 60:.1f} min)")

        append_metric_row(metric_path, {
            "epoch": epoch + 1,
            "train_box_loss": round(running["box"] / max(1, steps), 5),
            "train_obj_loss": round(running["obj"] / max(1, steps), 5),
            "train_cls_loss": round(running["cls"] / max(1, steps), 5),
            "val_loss": metrics["val_loss"],
            "val_box_loss": metrics["val_box_loss"],
            "val_obj_loss": metrics["val_obj_loss"],
            "val_cls_loss": metrics["val_cls_loss"],
            "map50": metrics["map50"], "map50_95": metrics["map50_95"],
            "tool_ap50": _ap50(per_class, "tool"), "tip_ap50": _ap50(per_class, "tip"),
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
                    "tip_box_size": args.tip_box_size,
                    "class_names": list(names), "label_set": args.label_set,
                    "dataset": args.dataset}, last_path)

        if fitness > best_fitness:
            best_fitness = fitness
            saved = build(num_classes=len(names), **arch)
            saved.load_state_dict(state)
            save_checkpoint(best_path, saved, arch, args.image_size, args.tip_box_size,
                            args.dataset, epoch + 1, metrics,
                            class_names=names, label_set=args.label_set)
            print(f"  saved {best_path} (mAP@0.5:0.95 {fitness:.4f})")

        with open(status_path, "w", encoding="utf-8") as handle:
            json.dump({"dataset": args.dataset, "epochs_completed": epoch + 1,
                       "epochs_total": args.epochs, "best_map50_95": best_fitness,
                       "last_metrics": metrics, "args": vars(args)}, handle, indent=2)

    print(f"done. best mAP@0.5:0.95 {best_fitness:.4f} -> {best_path}")


def _ap50(per_class: dict, name: str):
    """AP@0.5 of one class, or None when this run does not have that class.

    `tiponly` has no `tool` class at all, and a class the validation split
    happened to contain no instances of is absent too."""
    return per_class.get(name, {}).get("ap50")


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
