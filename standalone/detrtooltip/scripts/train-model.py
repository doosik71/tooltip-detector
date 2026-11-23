#!/usr/bin/env python3
"""Stage 2: train the whole detrtooltip model on the annotations.

The backbone starts from stage 1 (pretrain.pt) and is trained along with the
head, at `--backbone-lr-factor` of the head's learning rate. A head-only stage
before this was dropped: in short runs it trailed full training at every epoch
(docs/IMPLEMENTATION-PLAN.md, section 5.1).

`--init` overrides where the weights come from: a pretrain.pt carries over
only the backbone, a detector checkpoint everything (to continue training),
and `none` starts from random weights, for comparing against pretraining.
When the backbone comes from a checkpoint, its BatchNorm statistics stay as
they were; batches of a few 736 x 480 frames would only add noise to them.

Every split is used for training except the videos in `--holdout-videos`
(video42 by default), which pick the checkpoint: the best tip F1 @ 20 px over
`tip` and `tip2` (see common/evaluation.py). Frames are drawn with replacement,
weighted towards trusted labels and towards frames with a second tip
(common/dataset.py); an epoch is `--epoch-size` frames.

Outputs under standalone/detrtooltip/data/model/<dataset>/ by default:

    model.pt            best checkpoint (EMA weights)
    model-last.pt       last epoch, plus optimizer/EMA state for resuming
    metric.csv          one row per epoch
    train-status.json   epochs completed, best score, run arguments

Usage:
    ./standalone/detrtooltip/run train-model --dataset cholec80 --device cuda:1
"""

import argparse
import copy
import csv
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

if True:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from common.dataset import (DetectionDataset, available_datasets, collate,
                                default_data_root, list_frames, parse_label, sample_weights,
                                TRUSTED_SOURCES)
    from common.evaluation import Evaluation
    from common.inference import FAMILY, model_dir, save_checkpoint
    from common.loss import SetCriterion
    from common.model import DEFAULT_ARCH, build, parameter_count, postprocess
    from common.progress import progress

WARMUP_STEPS = 500
FINAL_LR_FACTOR = 0.01
GRAD_CLIP_NORM = 0.1
# Deformable DETR trains the sampling offsets at a tenth of the learning rate.
OFFSET_LR_FACTOR = 0.1
LOSS_PARTS = ("cls", "box", "giou", "tip", "tip_refined", "exist")


class ModelEMA:
    """Exponential moving average of the weights; evaluated and saved instead
    of the raw model."""

    def __init__(self, model, decay: float = 0.9998, warmup: int = 2000):
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


def lr_factor(step: int, total: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    progress_ = min(1.0, (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS))
    return FINAL_LR_FACTOR + (1 - FINAL_LR_FACTOR) * 0.5 * (1 + math.cos(math.pi * progress_))


def read_init(path: str) -> dict | None:
    if path == "none":
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("family") != FAMILY:
        raise SystemExit(f"{path} is not a detrtooltip checkpoint")
    return checkpoint


def init_arch(checkpoint: dict | None, args) -> dict:
    """A detector checkpoint fixes the whole architecture; a pretrain one only
    the ResNet depth."""
    if checkpoint is None:
        return {**DEFAULT_ARCH, "depth": args.depth or DEFAULT_ARCH["depth"],
                "num_queries": args.num_queries}
    if checkpoint.get("stage") == "pretrain":
        return {**DEFAULT_ARCH, "depth": checkpoint["arch"]["depth"],
                "num_queries": args.num_queries}
    return {**DEFAULT_ARCH, **checkpoint["arch"]}


def load_init(model, checkpoint: dict, path: str) -> str:
    """Load stage-2 starting weights; returns what was loaded, for the log."""
    if checkpoint.get("stage") == "pretrain":
        state = {key.removeprefix("backbone."): value
                 for key, value in checkpoint["model"].items() if key.startswith("backbone.")}
        model.backbone.load_state_dict(state)
        return f"backbone from {path} (pretrain epoch {checkpoint.get('epoch')})"
    model.load_state_dict(checkpoint["model"])
    return f"all weights from {path} (stage {checkpoint.get('stage')}, epoch {checkpoint.get('epoch')})"


def set_train_mode(model, freeze_backbone_bn: bool) -> None:
    model.train()
    if freeze_backbone_bn:
        for module in model.backbone.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()


def build_optimizer(model, lr: float, backbone_factor: float, weight_decay: float):
    groups = {"backbone": [], "offsets": [], "head": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            groups["backbone"].append(parameter)
        elif "sampling_offsets" in name:
            groups["offsets"].append(parameter)
        else:
            groups["head"].append(parameter)
    factors = {"backbone": backbone_factor, "offsets": OFFSET_LR_FACTOR, "head": 1.0}
    return torch.optim.AdamW([{"params": params, "lr": lr * factors[name], "name": name}
                              for name, params in groups.items() if params],
                             lr=lr, weight_decay=weight_decay)


def to_device(targets: list, device) -> list:
    return [{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
             for k, v in t.items()} for t in targets]


@torch.no_grad()
def select_score(model, loader, frames, conf: float, device) -> dict:
    """Score the held-out frames with the shared evaluation."""
    model.eval()
    evaluation = Evaluation(conf)
    for images, _, indices in progress(loader, desc="  select", leave=False):
        images = images.to(device, non_blocking=True).float().div_(255.0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(images)
        for predictions, index in zip(postprocess(outputs, 0.0), indices):
            frame = frames[index]
            evaluation.add(parse_label(frame.label), predictions,
                           frame.source in TRUSTED_SOURCES)
    return evaluation.compute()


def main():
    parser = argparse.ArgumentParser(description="Stage 2: train detrtooltip on annotations")
    parser.add_argument("--dataset", required=True, choices=available_datasets() or None)
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--init", default=None,
                        help="weights to start from: a pretrain.pt, a detector checkpoint or "
                             "'none' (default: data/model/<dataset>/pretrain.pt)")
    parser.add_argument("--holdout-videos", default="video42",
                        help="comma-separated videos kept out of training to pick the "
                             "checkpoint with ('' = train on every video, no selection)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--epoch-size", type=int, default=30000, help="frames per epoch")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1,
                        help="backbone lr = lr * this")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--two-tip-weight", type=float, default=10.0,
                        help="sampling weight of frames with a second tip")
    parser.add_argument("--trusted-weight", type=float, default=3.0,
                        help="sampling weight of other manual/opus frames (annotation/ = 1)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="score threshold for the tip metrics of checkpoint selection")
    parser.add_argument("--select-stride", type=int, default=1,
                        help="score every Nth held-out frame")
    parser.add_argument("--num-queries", type=int, default=DEFAULT_ARCH["num_queries"])
    parser.add_argument("--depth", type=int, default=None, choices=(50, 101),
                        help="ResNet depth with --init none (otherwise taken from --init)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base_dir = model_dir(args.dataset)
    if args.init is None:
        args.init = os.path.join(base_dir, "pretrain.pt")
    if args.init != "none" and not os.path.exists(args.init):
        raise SystemExit(f"--init not found: {args.init}")
    args.output_dir = args.output_dir or base_dir
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "model.pt")
    last_path = os.path.join(args.output_dir, "model-last.pt")
    metric_path = os.path.join(args.output_dir, "metric.csv")
    status_path = os.path.join(args.output_dir, "train-status.json")

    # The architecture follows the checkpoint we start from.
    init_checkpoint = read_init(args.init)
    arch = init_arch(init_checkpoint, args)
    model = build(**arch)
    loaded = (load_init(model, init_checkpoint, args.init) if init_checkpoint is not None
              else "random initialisation")
    del init_checkpoint
    model.to(device)

    freeze_backbone_bn = args.init != "none"
    print(f"detrtooltip  {parameter_count(model) / 1e6:.2f} M parameters  [{device}]")
    print(f"  init: {loaded}")

    holdout = {v.strip() for v in args.holdout_videos.split(",") if v.strip()}
    train_frames = list_frames(args.dataset, args.data_root, exclude_videos=holdout)
    select_frames = (list_frames(args.dataset, args.data_root, videos=holdout,
                                 frame_stride=args.select_stride) if holdout else [])
    select_frames = [f for f in select_frames if f.label is not None]
    weights = sample_weights(train_frames, args.two_tip_weight, args.trusted_weight)
    print(f"  train frames {len(train_frames):,} ({sum(f.two_tips for f in train_frames):,} "
          f"with a second tip), {args.epoch_size:,} per epoch; "
          f"selection frames {len(select_frames):,} from {', '.join(sorted(holdout)) or '-'}")

    sampler = WeightedRandomSampler(weights, num_samples=args.epoch_size, replacement=True)
    train_loader = DataLoader(DetectionDataset(train_frames, augment=True),
                              batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, collate_fn=collate, pin_memory=True,
                              drop_last=True, persistent_workers=args.workers > 0)
    select_loader = (DataLoader(DetectionDataset(select_frames, augment=False),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, collate_fn=collate, pin_memory=True)
                     if select_frames else None)

    criterion = SetCriterion()
    optimizer = build_optimizer(model, args.lr, args.backbone_lr_factor, args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total_steps))
    ema = None if args.no_ema else ModelEMA(model)

    start_epoch, best_score = 0, -1.0
    if not args.no_resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if ema is not None and state.get("ema") is not None:
            ema.ema.load_state_dict(state["ema"])
            ema.updates = state.get("ema_updates", 0)
        start_epoch, best_score = state["epoch"], state.get("best_score", -1.0)
        print(f"  resuming after epoch {start_epoch} (best {best_score:.4f})")

    for epoch in range(start_epoch, args.epochs):
        set_train_mode(model, freeze_backbone_bn)
        started = time.perf_counter()
        running = {key: 0.0 for key in LOSS_PARTS}
        steps = 0
        bar = progress(train_loader, desc=f"train {epoch + 1}/{args.epochs}")
        for images, targets, _ in bar:
            images = images.to(device, non_blocking=True).float().div_(255.0)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(images)
            loss, parts = criterion(outputs, to_device(targets, device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            for key in running:
                running[key] += parts[key]
            steps += 1
            bar.set_postfix(loss=f"{sum(running.values()) / steps:.3f}",
                            tip=f"{running['tip_refined'] / steps:.4f}")

        evaluated = ema.ema if ema is not None else model
        metrics = (select_score(evaluated, select_loader, select_frames, args.conf, device)
                   if select_loader else None)
        score = metrics["selection_tip_f1_20px"] if metrics else float(epoch)
        elapsed = time.perf_counter() - started

        row = {"epoch": epoch + 1,
               **{f"train_{k}": round(running[k] / max(1, steps), 5) for k in LOSS_PARTS}}
        if metrics:
            pairs, tips = metrics["pairs"], metrics["tips"]
            row.update({
                "tip_f1_20px": metrics["selection_tip_f1_20px"],
                "tool_ap50": metrics["tool_ap50"], "tool_ap50_95": metrics["tool_ap50_95"],
                "tip_hit_20px_pct": tips["hit_rate_20px_pct"],
                "pair_hit_10px_pct": pairs["pair_hit_rate_10px_pct"],
                "pair_hit_20px_pct": pairs["pair_hit_rate_20px_pct"],
                "two_tip_pair_hit_20px_pct": pairs["two_tip_pair_hit_rate_20px_pct"],
                "tip2_f1": pairs["tip2"]["f1"],
            })
            print(f"  epoch {epoch + 1}: tip F1@20 {row['tip_f1_20px']:.4f}  "
                  f"tool AP50 {row['tool_ap50']}  pair@20 {row['pair_hit_20px_pct']:.2f} %  "
                  f"two-tip pair@20 {row['two_tip_pair_hit_20px_pct']:.2f} %  "
                  f"tip2 F1 {row['tip2_f1']:.4f}  ({elapsed / 60:.1f} min)")
        else:
            print(f"  epoch {epoch + 1}: no held-out videos, last epoch kept "
                  f"({elapsed / 60:.1f} min)")
        head_lr = next(g["lr"] for g in optimizer.param_groups if g["name"] == "head")
        row.update({"lr": round(head_lr, 8), "seconds": round(elapsed, 1)})
        exists = os.path.exists(metric_path)
        with open(metric_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

        if score > best_score:
            best_score = score
            save_checkpoint(best_path, evaluated, arch, args.dataset, "train", epoch + 1,
                            metrics)
            print(f"  saved {best_path}")

        temporary = last_path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "ema": ema.ema.state_dict() if ema is not None else None,
                    "ema_updates": ema.updates if ema is not None else 0,
                    "epoch": epoch + 1, "best_score": best_score, "arch": arch}, temporary)
        os.replace(temporary, last_path)
        with open(status_path, "w", encoding="utf-8") as handle:
            json.dump({"dataset": args.dataset,
                       "epochs_completed": epoch + 1, "epochs_total": args.epochs,
                       "best_tip_f1_20px": best_score, "last_metrics": metrics,
                       "args": vars(args)}, handle, indent=2)

    print(f"done. best tip F1@20 {best_score:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
