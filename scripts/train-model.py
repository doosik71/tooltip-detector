"""Training script for TooltipDetector.

Usage:
    uv run python scripts/train-model.py --dataset cholec80 [--epochs N] [--batch-size N] [--lr F]
    uv run python scripts/train-model.py --dataset cholec80 --target-mode gaussian-tip --gaussian-sigma 15

Target modes (--target-mode)
-----------------------------
  gradient-seg  (default) — needs segmentation masks. Per-tool distance
                 gradient over the masked tool area.
  gaussian-tip            — needs only tip coordinates, no segmentation
                 masks. Gaussian centered on each tip (--gaussian-sigma px).

Checkpoints
-----------
  data/models/<dataset>/<target-mode>/<model-type>/best.pt         — lowest validation-loss model seen so far
  data/models/<dataset>/<target-mode>/<model-type>/last.pt         — model state at the end of the most recent epoch
  data/models/<dataset>/<target-mode>/<model-type>/train-status.json
      — progress record written alongside last.pt at the end of every epoch
        (completed epoch count, best val loss so far, run hyperparameters).
        Read back on resume so an interrupted run can continue from where
        it left off instead of restarting at epoch 1.
  data/models/<dataset>/<target-mode>/<model-type>/metric.csv
      — one row per epoch (train/val loss + pixel-wise heatmap-error mae/me/std,
        lr, timing) for plotting training curves across the whole run,
        including resumed sessions.

Resuming
--------
  Resuming is the default -- if train-status.json exists in <model-dir>,
  training continues from completed_epochs + 1 with best_val_loss and the
  LR schedule position restored (the optimizer's own momentum/variance
  state is not persisted and is reinitialized). --epochs on the resuming
  command is the new total epoch target -- it does not need to match the
  original run. If only last.pt exists (no train-status.json, e.g. a
  checkpoint from before this feature existed), only the weights are
  restored and the epoch count restarts at 1.
  Pass --no-resume to ignore any existing last.pt/train-status.json/metric.csv
  and train from scratch (existing files are overwritten as training proceeds).
"""

import argparse
import csv
import json
import os
import sys
import time

import albumentations as A
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ttd.checkpoints import model_dir as default_model_dir
from ttd.dataset import (DATASETS, DEFAULT_GAUSSIAN_SIGMA, DEFAULT_TARGET_MODE, TARGET_MODES,
                         SurgicalToolDataset, require_samples)
from ttd.model import REGISTRY as MODEL_REGISTRY
from ttd.model import build as build_model
from ttd.transforms import _eval_transform

# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------


def _train_transform(image_size=(480, 736)):
    """Geometric + photometric augmentation for training frames.

    All spatial transforms are applied to the image and the target mask
    simultaneously. Colour/noise transforms are image-only.
    """
    h, w = image_size
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.85, 1.15),
            rotate=(-20, 20),
            p=0.6,
        ),
        A.RandomResizedCrop(size=(h, w), scale=(0.7, 1.0),
                            ratio=(w / h * 0.9, w / h * 1.1), p=0.5),
        A.Resize(h, w),
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.2, hue=0.05, p=0.6),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def tip_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss on the tool channel (channel 1) against the distance-based target.

    pred:   (B, 2, H, W) raw logits from TooltipDetector
    target: (B, H, W) float32 heatmap in [0, 1]
    """
    return F.mse_loss(torch.sigmoid(pred[:, 1]), target)


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _fmt_time(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _save_train_status(
    path: str,
    args: argparse.Namespace,
    completed_epochs: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    elapsed_seconds: float,
) -> None:
    """Write the resume record read back on resume (see module docstring)."""
    status = {
        "dataset": args.dataset,
        "model_type": args.model_type,
        "target_mode": args.target_mode,
        "gaussian_sigma": args.gaussian_sigma,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "completed_epochs": completed_epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed_seconds,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w") as f:
        json.dump(status, f, indent=2)


_METRIC_FIELDS = [
    "epoch", "train_loss", "train_mae", "train_me", "train_std",
    "val_loss", "val_mae", "val_me", "val_std", "lr", "epoch_sec", "elapsed_sec",
]


def _append_metric_row(path: str, write_header: bool, row: dict) -> None:
    """Append one epoch's row to metric.csv, writing the header first if needed."""
    if write_header:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_METRIC_FIELDS).writeheader()
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=_METRIC_FIELDS).writerow(row)


# ---------------------------------------------------------------------------
# Training / validation epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> dict:
    """Run one train or val pass and return per-pixel heatmap-error stats.

    ``mae``/``me``/``std`` characterize sigmoid(pred[:, 1]) - target over every
    pixel of every frame in the loader: mean absolute error, mean (signed)
    error/bias, and the standard deviation of that error. They are cheap
    running sums alongside the existing loss, not a separate detection pass.
    """
    model.train(train)
    phase = "train" if train else "val"
    total_loss = 0.0
    total_n = 0
    sum_err = 0.0
    sum_abs_err = 0.0
    sum_sq_err = 0.0
    total_px = 0

    with torch.set_grad_enabled(train):
        pbar = tqdm(loader, desc=f"  [{phase}]", ascii=True, ncols=100)
        for images, targets in pbar:
            images = images.to(device,  dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            preds = model(images)
            loss = tip_loss(preds, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_n += images.size(0)
            pbar.set_postfix(loss=f"{total_loss / total_n:.6f}")

            with torch.no_grad():
                diff = torch.sigmoid(preds[:, 1]) - targets
                sum_err += diff.sum().item()
                sum_abs_err += diff.abs().sum().item()
                sum_sq_err += diff.square().sum().item()
                total_px += diff.numel()

    me = sum_err / total_px
    var = max(sum_sq_err / total_px - me * me, 0.0)
    return {
        "loss": total_loss / total_n,
        "mae": sum_abs_err / total_px,
        "me": me,
        "std": var ** 0.5,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASETS),
                        help="Dataset name under --data-root (e.g. cholec80)")
    parser.add_argument("--model-type", default="monai",
                        choices=list(MODEL_REGISTRY),
                        help="Model architecture to train (default: monai)")
    parser.add_argument("--target-mode", default=DEFAULT_TARGET_MODE,
                        choices=list(TARGET_MODES),
                        help="Training target generation method "
                             f"(default: {DEFAULT_TARGET_MODE})")
    parser.add_argument("--gaussian-sigma", type=float, default=DEFAULT_GAUSSIAN_SIGMA,
                        help="Gaussian std-dev in px, only used when "
                             f"--target-mode=gaussian-tip (default: {DEFAULT_GAUSSIAN_SIGMA})")
    parser.add_argument("--data-root",  default="data/dataset",
                        help="Root directory containing <dataset>/ subdirectories "
                             "(default: data/dataset)")
    parser.add_argument("--model-dir",  default=None,
                        help="Directory for best.pt and last.pt "
                             "(default: data/models/<dataset>/<target-mode>/<model-type>)")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--device",     default="",
                        help="torch device, e.g. 'cuda:1' to pick one of "
                             "several GPUs (default: cuda if available else cpu)")
    parser.add_argument("--no-resume",  action="store_true",
                        help="Ignore any existing last.pt/train-status.json/"
                             "metric.csv in <model-dir> and train from scratch "
                             "(resuming is the default when they exist)")
    args = parser.parse_args()
    resume = not args.no_resume

    if args.model_dir is None:
        args.model_dir = default_model_dir(
            model_type=args.model_type, dataset_name=args.dataset,
            target_mode=args.target_mode)

    best_path = os.path.join(args.model_dir, "best.pt")
    last_path = os.path.join(args.model_dir, "last.pt")
    status_path = os.path.join(args.model_dir, "train-status.json")
    metric_path = os.path.join(args.model_dir, "metric.csv")
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu")
    )

    # ── Datasets & loaders ───────────────────────────────────────────────
    dataset_root = os.path.join(args.data_root, args.dataset)
    train_ds = SurgicalToolDataset(
        dataset_root, "train", transform=_train_transform(),
        target_mode=args.target_mode, gaussian_sigma=args.gaussian_sigma)
    val_ds = SurgicalToolDataset(
        dataset_root, "val",   transform=_eval_transform(),
        target_mode=args.target_mode, gaussian_sigma=args.gaussian_sigma)
    require_samples(train_ds, "train", dataset_root)
    require_samples(val_ds, "val", dataset_root)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")
    prev_elapsed = 0.0

    model = build_model(args.model_type, num_classes=2).to(device)
    status = None
    if resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state)

        if os.path.exists(status_path):
            with open(status_path) as f:
                status = json.load(f)

        if status is not None:
            start_epoch = status["completed_epochs"] + 1
            best_val_loss = status["best_val_loss"]
            prev_elapsed = status.get("elapsed_seconds", 0.0)
            print(f"Resumed from {last_path}  "
                  f"({status['completed_epochs']} epochs completed, "
                  f"best_val_loss={best_val_loss:.6f})")
        else:
            print(f"Resumed weights from {last_path}  "
                  f"(no {os.path.basename(status_path)} found, "
                  "epoch count restarts at 1)")

    # Fresh run (or legacy weights-only resume): metric.csv is rewritten from
    # scratch. A true resume (status found above) appends to the existing file.
    metric_needs_header = status is None or not os.path.exists(metric_path)

    # optimizer/scheduler are always freshly initialized (Adam momentum is not
    # persisted across --resume); `last_epoch` below fast-forwards the cosine
    # schedule to where a resumed run left off without needing saved state.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, last_epoch=start_epoch - 2)

    # ── Header ───────────────────────────────────────────────────────────
    print(f"Device     : {device}")
    print(f"Dataset    : {args.dataset}")
    print(f"Model type : {args.model_type}")
    print(f"Target mode: {args.target_mode}"
          + (f"  (gaussian_sigma={args.gaussian_sigma})" if args.target_mode == "gaussian-tip" else ""))
    print(
        f"Train      : {len(train_ds):,} samples  ({len(train_loader)} batches)")
    print(f"Val        : {len(val_ds):,} samples  ({len(val_loader)} batches)")
    print(
        f"Epochs     : {args.epochs}   batch={args.batch_size}   lr={args.lr:.2e}")
    print(f"Checkpoints: {best_path}  /  {last_path}")
    print(f"Metrics    : {metric_path}")
    print()

    if start_epoch > args.epochs:
        print(f"Nothing to do: already completed {start_epoch - 1}/{args.epochs} "
              f"epochs (best_val_loss={best_val_loss:.6f}). "
              "Pass a larger --epochs to keep training.")
        return

    train_start = time.time()

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs}  lr={lr_now:.2e}")

        epoch_t0 = time.time()
        train_stats = _run_epoch(model, train_loader,
                                 optimizer, device, train=True)
        val_stats = _run_epoch(model, val_loader,
                               optimizer, device, train=False)
        scheduler.step()

        train_loss, val_loss = train_stats["loss"], val_stats["loss"]
        epoch_time = time.time() - epoch_t0
        total_elapsed = prev_elapsed + (time.time() - train_start)
        eta_total = total_elapsed / epoch * (args.epochs - epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        star = " ★" if is_best else "  "

        print(
            f"  {star} train={train_loss:.6f}  val={val_loss:.6f}"
            f"  epoch={_fmt_time(epoch_time)}"
            f"  elapsed={_fmt_time(total_elapsed)}"
            f"  eta={_fmt_time(eta_total)}"
        )

        # ── Checkpoints ──────────────────────────────────────────────────
        torch.save(model.state_dict(), last_path)
        _save_train_status(
            status_path, args, completed_epochs=epoch,
            train_loss=train_loss, val_loss=val_loss,
            best_val_loss=best_val_loss, elapsed_seconds=total_elapsed,
        )
        _append_metric_row(metric_path, metric_needs_header, {
            "epoch": epoch,
            "train_loss": train_loss, "train_mae": train_stats["mae"],
            "train_me": train_stats["me"], "train_std": train_stats["std"],
            "val_loss": val_loss, "val_mae": val_stats["mae"],
            "val_me": val_stats["me"], "val_std": val_stats["std"],
            "lr": lr_now, "epoch_sec": round(epoch_time, 2),
            "elapsed_sec": round(total_elapsed, 2),
        })
        metric_needs_header = False

        if is_best:
            torch.save(model.state_dict(), best_path)
            print(f"    best.pt updated  (val_loss={best_val_loss:.6f})")

        print()

    print(
        f"Training complete.  Total time: {_fmt_time(total_elapsed)}"
        f"  (this session: {_fmt_time(time.time() - train_start)})")
    print(f"Best val loss : {best_val_loss:.6f}  → {best_path}")
    print(f"Last model    : {last_path}")


if __name__ == "__main__":
    main()
