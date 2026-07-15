"""Training script for TooltipDetector.

Usage:
    uv run python -m ttd.train [--epochs N] [--batch-size N] [--lr F] [--data-root PATH]

Checkpoints
-----------
  data/model/best.pt  — lowest validation-loss model seen so far
  data/model/last.pt  — model state at the end of the most recent epoch
"""

import argparse
import os
import time

import albumentations as A
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from ttd.dataset import SurgicalToolDataset
from ttd.model import REGISTRY as MODEL_REGISTRY
from ttd.model import build as build_model

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


def _eval_transform(image_size=(480, 736)):
    h, w = image_size
    return A.Compose([
        A.Resize(h, w),
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

_BAR_WIDTH = 30


def _fmt_time(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _bar(step: int, total: int) -> str:
    filled = int(_BAR_WIDTH * step / total)
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


# ---------------------------------------------------------------------------
# Training / validation epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> float:
    model.train(train)
    phase = "train" if train else "val  "
    total_loss = 0.0
    total_n = 0
    n_steps = len(loader)
    t0 = time.time()

    with torch.set_grad_enabled(train):
        for step, (images, targets) in enumerate(loader, 1):
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

            elapsed = time.time() - t0
            eta = elapsed / step * (n_steps - step)
            avg = total_loss / total_n

            print(
                f"\r  [{phase}] |{_bar(step, n_steps)}|"
                f" {step:4d}/{n_steps}"
                f"  loss={avg:.6f}"
                f"  {_fmt_time(elapsed)}<{_fmt_time(eta)}",
                end="", flush=True,
            )

    # Clear the progress line
    print()
    return total_loss / total_n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="monai",
                        choices=list(MODEL_REGISTRY),
                        help="Model architecture to train (default: monai)")
    parser.add_argument("--data-root",  default="data/dataset")
    parser.add_argument("--model-dir",  default=None,
                        help="Directory for best.pt and last.pt "
                             "(default: data/models/<model-type>)")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from <model-dir>/last.pt if it exists")
    args = parser.parse_args()

    if args.model_dir is None:
        args.model_dir = f"data/models/{args.model_type}"

    best_path = os.path.join(args.model_dir, "best.pt")
    last_path = os.path.join(args.model_dir, "last.pt")
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Datasets & loaders ───────────────────────────────────────────────
    train_ds = SurgicalToolDataset(
        args.data_root, "train", transform=_train_transform())
    val_ds = SurgicalToolDataset(
        args.data_root, "val",   transform=_eval_transform())

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(args.model_type, num_classes=2).to(device)
    if args.resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        print(f"Resumed from {last_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ── Header ───────────────────────────────────────────────────────────
    print(f"Device     : {device}")
    print(f"Model type : {args.model_type}")
    print(
        f"Train      : {len(train_ds):,} samples  ({len(train_loader)} batches)")
    print(f"Val        : {len(val_ds):,} samples  ({len(val_loader)} batches)")
    print(
        f"Epochs     : {args.epochs}   batch={args.batch_size}   lr={args.lr:.2e}")
    print(f"Checkpoints: {best_path}  /  {last_path}")
    print()

    best_val_loss = float("inf")
    train_start = time.time()

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs}  lr={lr_now:.2e}")

        epoch_t0 = time.time()
        train_loss = _run_epoch(model, train_loader,
                                optimizer, device, train=True)
        val_loss = _run_epoch(model, val_loader,
                              optimizer, device, train=False)
        scheduler.step()

        epoch_time = time.time() - epoch_t0
        total_elapsed = time.time() - train_start
        eta_total = total_elapsed / epoch * (args.epochs - epoch)

        is_best = val_loss < best_val_loss
        star = " ★" if is_best else "  "

        print(
            f"  {star} train={train_loss:.6f}  val={val_loss:.6f}"
            f"  epoch={_fmt_time(epoch_time)}"
            f"  elapsed={_fmt_time(total_elapsed)}"
            f"  eta={_fmt_time(eta_total)}"
        )

        # ── Checkpoints ──────────────────────────────────────────────────
        torch.save(model.state_dict(), last_path)

        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"    best.pt updated  (val_loss={best_val_loss:.6f})")

        print()

    print(
        f"Training complete.  Total time: {_fmt_time(time.time() - train_start)}")
    print(f"Best val loss : {best_val_loss:.6f}  → {best_path}")
    print(f"Last model    : {last_path}")


if __name__ == "__main__":
    main()
