#!/usr/bin/env python3
"""Train the YOLO26 clone on this repository's surgical-tool annotations.

Two classes are learned from one annotation file (see common/dataset.py):
`tool` (the annotated bounding box) and `tip` (a square box of
`--tip-box-size` px centred on the annotated tool tip), so one detector
produces both the instrument box and the tip coordinate the root project's
metrics are defined on.

What is YOLO26's here, and reimplemented as such: the architecture, the
end-to-end head with its two branches, the DFL-free box loss, the assigner's
box floor and second top-k, and MuSGD, the hybrid Muon/SGD optimizer
`optimizer=auto` resolves to in the reference.

What is deliberately *not* YOLO26's: the outer training loop. 640 x 640,
mosaic, a three-epoch linear warmup then cosine decay, a weight EMA
(`--no-ema` to disable) and no gradient accumulation -- exactly the loop the
CLAD-Net and YOLOv8s-clone baselines use. Keeping it identical is what makes
the three reimplementations comparable; the differences between them are then
differences of model and loss, not of bookkeeping. `--optimizer sgd` extends
that to the optimizer.

No pretrained weights are used. The reference checkpoint is COCO-trained at 80
classes and reading it needs `ultralytics`; it is used to verify this
reimplementation (scripts/verify-clone.py), never to initialise it.

Outputs, all under baseline/yolo26clone/data/model/<dataset>/ by default:

    model.pt            best checkpoint by val mAP@0.5:0.95
    model-last.pt       last epoch, plus optimizer/EMA state for --resume
    train-status.json   epochs completed, best fitness, run arguments
    metric.csv          one row per epoch (losses, mAP, lr, time)

Usage:
    ./baseline/yolo26clone/run train-model --dataset cholec80
    ./baseline/yolo26clone/run train-model --dataset cholec80 --optimizer sgd --device cuda:1
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

    from common.boxes import xywh_to_xyxy
    from common.dataset import (CLASS_NAMES, DEFAULT_TIP_BOX_SIZE, SurgicalDetectionDataset,
                                available_datasets, collate, default_data_root)
    from common.inference import model_dir, save_checkpoint
    from common.loss import DetectionLoss
    from common.metrics import DetectionEvaluator
    from common.optim import build_optimizer
    from common.progress import progress
    from common.model import YOLO26, build, decode, parameter_count, postprocess

WARMUP_EPOCHS = 3.0
FINAL_LR_FACTOR = 0.01      # cosine decays lr to lr * this
GRAD_CLIP_NORM = 10.0       # matches the reference trainer


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


def float_predictions(preds: dict) -> dict:
    """Cast the head's outputs to fp32 for the loss.

    Only `boxes` and `scores` are cast. `feats` is read for its shapes and for
    the anchor grid, whose coordinates are half-integers below 100 -- exactly
    representable in fp16, so there is nothing to gain from copying whole
    feature maps.
    """
    return {branch: {"boxes": out["boxes"].float(), "scores": out["scores"].float(),
                     "feats": out["feats"]}
            for branch, out in preds.items()}


def lr_lambda(epoch: float, epochs: int) -> float:
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1e-9) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1e-9, epochs - WARMUP_EPOCHS)
    return FINAL_LR_FACTOR + (1 - FINAL_LR_FACTOR) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def validate(model, loader, criterion, device, conf: float, max_det: int) -> dict:
    model.eval()
    evaluator = DetectionEvaluator(len(CLASS_NAMES), CLASS_NAMES)
    totals = {"box": 0.0, "cls": 0.0, "l1": 0.0}
    batches = 0

    for images, targets, _ in progress(loader, desc="  val", leave=False):
        images = images.to(device, non_blocking=True).float().div_(255.0)
        preds = float_predictions(model(images))
        _, parts = criterion(preds, targets)
        for key in totals:
            totals[key] += parts[key]
        batches += 1

        # Only the one-to-one branch is scored: it is the branch inference
        # reads, and the only one that needs no NMS behind it.
        detections = postprocess(decode(preds["one2one"]), max_det)
        size = images.shape[-1]
        for i, detection in enumerate(detections):
            detection = detection[detection[:, 4] >= conf]
            rows = targets[targets[:, 0] == i]
            ground_truth = np.zeros((len(rows), 5), dtype=np.float32)
            if len(rows):
                ground_truth[:, 0] = rows[:, 1].numpy()
                ground_truth[:, 1:] = xywh_to_xyxy(rows[:, 2:] * size).numpy()
            evaluator.add(detection.cpu().numpy(), ground_truth)

    metrics = evaluator.compute()
    batches = max(1, batches)
    metrics["val_box_loss"] = round(totals["box"] / batches, 5)
    metrics["val_cls_loss"] = round(totals["cls"] / batches, 5)
    metrics["val_l1_loss"] = round(totals["l1"] / batches, 5)
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
        description="Train the YOLO26 clone on a tooltip-detector dataset")
    parser.add_argument("--dataset", required=True, choices=available_datasets() or None,
                        help="dataset directory under data/dataset (e.g. cholec80)")
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--epochs", type=int, default=150,
                        help="(default: 150)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", default="musgd", choices=("musgd", "sgd"),
                        help="musgd (default) is what YOLO26 trains with; sgd matches "
                             "the CLAD-Net and YOLOv8s-clone baselines")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="0.9 is MuSGD's; the SGD baselines use 0.937")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="use every Nth train frame; consecutive video frames are near-duplicates")
    parser.add_argument("--val-frames", type=int, default=2000,
                        help="cap on validation frames per epoch (0 = all)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None,
                        help="where the checkpoints go (default: data/model/<dataset>)")
    parser.add_argument("--scale", default="s", choices=tuple(YOLO26.SCALES),
                        help="YOLO26 depth/width scale (default: s, the size the YOLOv8s "
                             "baselines use)")
    parser.add_argument("--tip-box-size", type=float, default=DEFAULT_TIP_BOX_SIZE,
                        help="side of the square box drawn around each annotated tip, in "
                             f"original-frame px (default: {DEFAULT_TIP_BOX_SIZE:g})")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="confidence threshold used when scoring validation mAP")
    parser.add_argument("--max-det", type=int, default=300,
                        help="cap on boxes per image when scoring validation. YOLO26's head "
                             "is end-to-end, so this replaces the NMS IoU threshold")
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
    model = build(num_classes=len(CLASS_NAMES), **arch).to(device)
    print(f"YOLO26{args.scale} clone  {parameter_count(model) / 1e6:.3f} M parameters  "
          f"[{device}]  tip box {args.tip_box_size:g} px  optimizer {args.optimizer}")

    train_set = SurgicalDetectionDataset(args.dataset, "train", args.image_size, augment=True,
                                         data_root=args.data_root, frame_stride=args.frame_stride,
                                         tip_box_size=args.tip_box_size)
    val_set = SurgicalDetectionDataset(args.dataset, "val", args.image_size, augment=False,
                                       data_root=args.data_root, limit=args.val_frames or None,
                                       tip_box_size=args.tip_box_size)
    print(f"train frames: {len(train_set):,}   val frames: {len(val_set):,}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate, pin_memory=True)

    criterion = DetectionLoss(model, epochs=args.epochs)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.momentum,
                                args.weight_decay)
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
        # The branch-blend schedule is a function of completed epochs, so it is
        # restored rather than restarted: a resumed epoch is weighted exactly
        # as it would have been in an uninterrupted run.
        criterion.step_epoch(start_epoch)
        print(f"resuming from epoch {start_epoch} (best fitness {best_fitness:.4f}, "
              f"one2many weight {criterion.o2m_weight:.3f})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.perf_counter()
        running = {"box": 0.0, "cls": 0.0, "l1": 0.0}
        steps = 0

        bar = progress(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for images, targets, _ in bar:
            images = images.to(device, non_blocking=True).float().div_(255.0)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
            # The loss runs in fp32 even under AMP. In half precision the CIoU
            # term's box-area products overflow and the whole loss goes NaN
            # within ~20 steps; the conv-heavy forward above keeps the speedup.
            loss, parts = criterion(float_predictions(outputs), targets)
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
            bar.set_postfix(box=f"{running['box'] / steps:.4f}",
                                 cls=f"{running['cls'] / steps:.4f}",
                                 l1=f"{running['l1'] / steps:.4f}")

        scheduler.step()
        # Shift the blend between the two head branches one epoch further
        # towards the one-to-one branch (see common/loss.py).
        criterion.step_epoch(epoch + 1)
        evaluated = ema.ema if ema is not None else model
        metrics = validate(evaluated, val_loader, criterion, device, args.conf, args.max_det)
        fitness = metrics["map50_95"] if metrics["map50_95"] is not None else -1.0
        elapsed = time.perf_counter() - epoch_start

        per_class = metrics["per_class"]
        print(f"  epoch {epoch + 1}: val_loss {metrics['val_loss']:.4f}  "
              f"mAP@0.5 {_fmt(metrics['map50'])}  mAP@0.5:0.95 {_fmt(metrics['map50_95'])}  "
              f"tool AP50 {_fmt(per_class['tool']['ap50'])}  tip AP50 {_fmt(per_class['tip']['ap50'])}  "
              f"({elapsed / 60:.1f} min)")

        append_metric_row(metric_path, {
            "epoch": epoch + 1,
            "train_box_loss": round(running["box"] / max(1, steps), 5),
            "train_cls_loss": round(running["cls"] / max(1, steps), 5),
            "train_l1_loss": round(running["l1"] / max(1, steps), 5),
            "val_loss": metrics["val_loss"],
            "val_box_loss": metrics["val_box_loss"],
            "val_cls_loss": metrics["val_cls_loss"],
            "val_l1_loss": metrics["val_l1_loss"],
            "map50": metrics["map50"], "map50_95": metrics["map50_95"],
            "tool_ap50": per_class["tool"]["ap50"], "tip_ap50": per_class["tip"]["ap50"],
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
                    "class_names": list(CLASS_NAMES), "dataset": args.dataset}, last_path)

        if fitness > best_fitness:
            best_fitness = fitness
            saved = build(num_classes=len(CLASS_NAMES), **arch)
            saved.load_state_dict(state)
            save_checkpoint(best_path, saved, arch, args.image_size, args.tip_box_size,
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
