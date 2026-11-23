#!/usr/bin/env python3
"""Stage 1: pretrain the detrtooltip backbone by masked reconstruction.

SimMIM-style: 60 % of the 32 x 32 patches of each frame are replaced by a
learned value, and the ResNet + FPN backbone with a light head predicts the
hidden pixels, scored against per-patch normalised targets (loss.py). No
labels are needed, so every frame of every split is used.

A fixed set of `--monitor-frames` frames, with fixed masks and no
augmentation, is scored after every epoch, and one of them is saved as an
image (original, masked input, reconstruction) to look at.

Outputs under standalone/detrtooltip/data/model/<dataset>/ by default:

    pretrain.pt              backbone + head, plus optimizer state for resuming
    pretrain-metric.csv      one row per epoch
    pretrain-sample.jpg      the monitor frame after the last epoch

Usage:
    ./standalone/detrtooltip/run pretrain-model --dataset cholec80 --device cuda:1
"""

import argparse
import csv
import math
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

if True:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from common.dataset import PretrainDataset, available_datasets, default_data_root, list_frames
    from common.inference import FAMILY, model_dir
    from common.loss import ReconstructionLoss, random_patch_mask
    from common.model import DEFAULT_ARCH, PIXEL_MEAN, PIXEL_STD, Pretrainer, parameter_count
    from common.progress import progress

WARMUP_STEPS = 1000
FINAL_LR_FACTOR = 0.01
GRAD_CLIP_NORM = 3.0


def lr_factor(step: int, total: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    progress_ = min(1.0, (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS))
    return FINAL_LR_FACTOR + (1 - FINAL_LR_FACTOR) * 0.5 * (1 + math.cos(math.pi * progress_))


@torch.no_grad()
def monitor(model, images, masks, criterion, sample_path: str) -> float:
    model.eval()
    losses = []
    for i in range(0, len(images), 16):
        batch, mask = images[i:i + 16], masks[i:i + 16]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=batch.is_cuda):
            prediction = model(batch, mask)
        normalised = model.backbone.normalise(batch)
        losses.append(float(criterion(prediction, normalised, mask)) * len(batch))
        if i == 0:
            save_sample(model, batch[0], mask[0], prediction[0].float(), normalised[0],
                        criterion, sample_path)
    model.train()
    return sum(losses) / len(images)


def save_sample(model, image, mask, prediction, normalised, criterion, path: str) -> None:
    """Original | masked input | reconstruction, the last de-normalised with
    each patch's own statistics so it is comparable to the original."""
    p = criterion.patch
    c, h, w = normalised.shape
    x = normalised.view(c, h // p, p, w // p, p)
    mean = x.mean(dim=(2, 4), keepdim=True)
    std = (x.var(dim=(2, 4), keepdim=True, unbiased=False) + 1e-6).sqrt()
    restored = (prediction.view(c, h // p, p, w // p, p) * std + mean).view(c, h, w)
    restored = restored * mask + normalised * (1 - mask)
    pixel_mean = torch.tensor(PIXEL_MEAN, device=image.device).view(3, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, device=image.device).view(3, 1, 1)
    restored = (restored * pixel_std + pixel_mean).clamp(0, 1)
    masked = image * (1 - mask) + 0.5 * mask
    row = torch.cat([image, masked, restored], dim=2)
    array = (row.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Stage 1: masked-reconstruction pretraining")
    parser.add_argument("--dataset", required=True, choices=available_datasets() or None)
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--epoch-size", type=int, default=60000,
                        help="frames drawn per epoch, without replacement (0 = all frames)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument("--depth", type=int, default=DEFAULT_ARCH["depth"], choices=(50, 101))
    parser.add_argument("--monitor-frames", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir = args.output_dir or model_dir(args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "pretrain.pt")
    metric_path = os.path.join(args.output_dir, "pretrain-metric.csv")
    sample_path = os.path.join(args.output_dir, "pretrain-sample.jpg")

    frames = list_frames(args.dataset, args.data_root)
    dataset = PretrainDataset(frames, augment=True)
    epoch_size = args.epoch_size or len(dataset)
    sampler = RandomSampler(dataset, replacement=False, num_samples=epoch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)

    arch = {"depth": args.depth}
    model = Pretrainer(**arch).to(device)
    criterion = ReconstructionLoss(args.patch)
    print(f"Pretrainer (ResNet-{args.depth} + FPN)  {parameter_count(model) / 1e6:.2f} M "
          f"parameters  [{device}]  frames {len(frames):,}, {epoch_size:,} per epoch")

    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and "mask_token" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1 or "mask_token" in n]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total_steps))

    # Fixed monitor frames and masks, spread over the whole frame list.
    picks = torch.linspace(0, len(frames) - 1, args.monitor_frames).long().tolist()
    monitor_set = PretrainDataset([frames[i] for i in picks], augment=False)
    monitor_images = torch.stack([monitor_set[i] for i in range(len(monitor_set))])
    monitor_images = monitor_images.to(device).float().div_(255.0)
    h, w = monitor_images.shape[-2:]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        monitor_masks = random_patch_mask(len(monitor_images), h, w, args.patch,
                                          args.mask_ratio, "cpu").to(device)

    start_epoch = 0
    if not args.no_resume and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        print(f"resuming after epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()
        running, steps = 0.0, 0
        bar = progress(loader, desc=f"pretrain {epoch + 1}/{args.epochs}")
        for images in bar:
            images = images.to(device, non_blocking=True).float().div_(255.0)
            mask = random_patch_mask(len(images), images.shape[2], images.shape[3], args.patch,
                                     args.mask_ratio, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(images, mask)
            loss = criterion(prediction, model.backbone.normalise(images), mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            running += float(loss)
            steps += 1
            bar.set_postfix(loss=f"{running / steps:.4f}")

        monitor_loss = monitor(model, monitor_images, monitor_masks, criterion, sample_path)
        elapsed = time.perf_counter() - started
        print(f"  epoch {epoch + 1}: train {running / max(1, steps):.4f}  "
              f"monitor {monitor_loss:.4f}  ({elapsed / 60:.1f} min)")
        row = {"epoch": epoch + 1, "train_loss": round(running / max(1, steps), 5),
               "monitor_loss": round(monitor_loss, 5),
               "lr": round(optimizer.param_groups[0]["lr"], 7), "seconds": round(elapsed, 1)}
        exists = os.path.exists(metric_path)
        with open(metric_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

        temporary = checkpoint_path + ".tmp"
        torch.save({"family": FAMILY, "stage": "pretrain", "arch": arch,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1,
                    "dataset": args.dataset, "args": vars(args)}, temporary)
        os.replace(temporary, checkpoint_path)

    print(f"done -> {checkpoint_path}")


if __name__ == "__main__":
    main()
