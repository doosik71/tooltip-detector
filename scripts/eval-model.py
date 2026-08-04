"""Evaluate tip-prediction accuracy on the test set.

The model outputs a 2-channel map.  Channel 1 (tool channel), after sigmoid,
is treated as a distance-based heatmap.  Peaks above a configurable threshold
are taken as candidate tip locations.  For each GT tip the nearest candidate
is selected and the Euclidean pixel distance is reported.

Results are saved to  data/results/<target-mode>/<model-type>/
  summary.json  — overall and per-session metrics + run parameters
  per_tip.csv   — one row per GT tip (coordinates, matched prediction, distance)

Metrics reported
----------------
  - Total GT tips and missed tips (no candidate found)
  - Mean / median distance (px) over matched tips
  - Hit-rate @ 10 / 20 / 50 px
  - Per-session breakdown

Usage
-----
    uv run python scripts/eval-model.py
    uv run python scripts/eval-model.py --threshold 0.4 --nms-radius 15
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ttd.checkpoints import default_model_path, default_results_dir
from ttd.dataset import DEFAULT_TARGET_MODE, TARGET_MODES, SurgicalToolDataset
from ttd.model import REGISTRY as MODEL_REGISTRY
from ttd.model import build as build_model
from ttd.peaks import find_peaks
from ttd.transforms import _eval_transform

_HIT_THRESHOLDS = (10, 20, 50)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _session_id(ann_path: str) -> str:
    """Extract session prefix from annotation path."""
    name = os.path.basename(ann_path)
    parts = name.rsplit("_", 1)
    return parts[0]


def evaluate(
    model_path: str,
    model_type: str,
    target_mode: str,
    data_root: str,
    threshold: float,
    nms_radius: int,
    batch_size: int,
    workers: int,
    device_str: str,
    results_root: str,
) -> dict:
    device = torch.device(
        device_str if device_str else (
            "cuda" if torch.cuda.is_available() else "cpu")
    )

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    results_dir = results_root
    os.makedirs(results_dir, exist_ok=True)

    print(f"Device      : {device}")
    print(f"Model type  : {model_type}")
    print(f"Target mode : {target_mode}")
    print(f"Model       : {model_path}")
    print(f"Threshold   : {threshold}   NMS radius: {nms_radius} px")
    print(f"Results dir : {results_dir}")

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(model_type, num_classes=2).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    # ── Dataset ──────────────────────────────────────────────────────────
    # target_mode only controls whether a segmentation mask is loaded per
    # frame; the GT tips used below are read directly from the annotation
    # JSON, not from the dataset's generated target heatmap.
    ds = SurgicalToolDataset(data_root, "test", transform=_eval_transform(),
                              target_mode=target_mode)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Test frames : {len(ds)}\n")

    # ── Accumulators ─────────────────────────────────────────────────────
    all_dists: list[float] = []
    n_gt_tips = 0
    n_missed = 0
    n_frames_with_tools = 0
    hits = {t: 0 for t in _HIT_THRESHOLDS}

    sess_dists: dict[str, list[float]] = defaultdict(list)
    sess_gt: dict[str, int] = defaultdict(int)
    sess_missed: dict[str, int] = defaultdict(int)

    tip_rows: list[dict] = []

    # ── Inference loop ───────────────────────────────────────────────────
    with torch.no_grad():
        for batch_i, (images, _) in enumerate(loader):
            images = images.to(device, dtype=torch.float32)
            preds = model(images)
            heatmaps = torch.sigmoid(preds[:, 1]).cpu().numpy()  # (B, H, W)

            for j in range(len(images)):
                sample_idx = batch_i * batch_size + j
                ann_path = ds.samples[sample_idx]
                stem = os.path.splitext(os.path.basename(ann_path))[0]

                with open(ann_path) as f:
                    ann_data = json.load(f)

                gt_tips = [
                    (a["tip"]["x"], a["tip"]["y"])
                    for a in ann_data["annotations"]
                ]
                if not gt_tips:
                    continue

                n_frames_with_tools += 1
                session = _session_id(ann_path)
                candidates = find_peaks(heatmaps[j], threshold, nms_radius)

                for gx, gy in gt_tips:
                    n_gt_tips += 1
                    sess_gt[session] += 1

                    if not candidates:
                        n_missed += 1
                        sess_missed[session] += 1
                        tip_rows.append({
                            "frame":    stem,
                            "session":  session,
                            "gt_x":     gx,
                            "gt_y":     gy,
                            "pred_x":   "",
                            "pred_y":   "",
                            "dist_px":  "",
                            "missed":   1,
                        })
                        continue

                    best = min(candidates,
                               key=lambda c: np.hypot(gx - c[0], gy - c[1]))
                    cx, cy = int(best[0]), int(best[1])
                    dist = float(np.hypot(gx - cx, gy - cy))

                    all_dists.append(dist)
                    sess_dists[session].append(dist)
                    for t in _HIT_THRESHOLDS:
                        if dist <= t:
                            hits[t] += 1

                    tip_rows.append({
                        "frame":    stem,
                        "session":  session,
                        "gt_x":     gx,
                        "gt_y":     gy,
                        "pred_x":   cx,
                        "pred_y":   cy,
                        "dist_px":  round(dist, 2),
                        "missed":   0,
                    })

            done = min((batch_i + 1) * batch_size, len(ds))
            if done % 5000 < batch_size or done == len(ds):
                print(f"  {done:>6}/{len(ds)}  "
                      f"gt_tips_seen={n_gt_tips}  "
                      f"missed={n_missed}")

    # ── Summary stats ────────────────────────────────────────────────────
    arr = np.array(all_dists) if all_dists else np.zeros(0)
    safe_n = max(1, n_gt_tips)

    stats: dict = {
        "timestamp":            run_ts,
        "model_type":           model_type,
        "target_mode":          target_mode,
        "model_path":           model_path,
        "threshold":            threshold,
        "nms_radius":           nms_radius,
        "data_root":            data_root,
        "n_frames_with_tools":  n_frames_with_tools,
        "n_gt_tips":            n_gt_tips,
        "n_missed":             n_missed,
        "miss_rate_pct":        round(n_missed / safe_n * 100, 2),
        "mean_dist_px":         round(float(arr.mean()),              2) if arr.size else None,
        "median_dist_px":       round(float(np.median(arr)),          2) if arr.size else None,
        "p90_dist_px":          round(float(np.percentile(arr, 90)),  2) if arr.size else None,
        **{f"hit_rate_{t}px_pct": round(hits[t] / safe_n * 100, 2)
           for t in _HIT_THRESHOLDS},
        "per_session": {
            sid: {
                "n_gt_tips":    sess_gt[sid],
                "n_missed":     sess_missed[sid],
                "mean_dist_px": round(float(np.mean(sess_dists[sid])), 2)
                if sess_dists[sid] else None,
            }
            for sid in sorted(sess_gt)
        },
    }

    _print_results(stats)
    _save_results(stats, tip_rows, results_dir)

    return stats


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

_CSV_FIELDS = ["frame", "session", "gt_x", "gt_y",
               "pred_x", "pred_y", "dist_px", "missed"]


def _save_results(stats: dict, tip_rows: list[dict], results_dir: str) -> None:
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(results_dir, "per_tip.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(tip_rows)

    print(f"\n  summary.json  → {summary_path}")
    print(f"  per_tip.csv   → {csv_path}  ({len(tip_rows):,} rows)")


# ---------------------------------------------------------------------------
# Formatted output
# ---------------------------------------------------------------------------

def _print_results(s: dict) -> None:
    W = 58

    def row(label, value):
        print(f"  {label:<34} {value}")

    print()
    print("=" * W)
    print(f"  Evaluation Results")
    print("=" * W)
    row("Model",            s["model_path"])
    row("Target mode",      s["target_mode"])
    row("Threshold / NMS radius",
        f"{s['threshold']}  /  {s['nms_radius']} px")
    print("-" * W)
    row("Frames with tools",       f"{s['n_frames_with_tools']:>10,}")
    row("GT tips total",            f"{s['n_gt_tips']:>10,}")
    row("Missed (no prediction)",
        f"{s['n_missed']:>10,}  ({s['miss_rate_pct']:.1f}%)")
    print("-" * W)
    mean = f"{s['mean_dist_px']:.2f} px" if s['mean_dist_px'] is not None else "n/a"
    med = f"{s['median_dist_px']:.2f} px" if s['median_dist_px'] is not None else "n/a"
    p90 = f"{s['p90_dist_px']:.2f} px" if s['p90_dist_px'] is not None else "n/a"
    row("Mean distance",   f"{mean:>14}")
    row("Median distance", f"{med:>14}")
    row("P90 distance",    f"{p90:>14}")
    print("-" * W)
    for t in _HIT_THRESHOLDS:
        row(f"Hit-rate @ {t:>2} px",
            f"{s[f'hit_rate_{t}px_pct']:>10.1f} %")
    print("-" * W)
    print("  Per-session:")
    for sid, ss in s["per_session"].items():
        md = f"{ss['mean_dist_px']:.1f}" if ss["mean_dist_px"] is not None else "n/a"
        print(f"    {sid}  "
              f"tips={ss['n_gt_tips']:>5}  "
              f"missed={ss['n_missed']:>4}  "
              f"mean={md} px")
    print("=" * W)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TooltipDetector tip-prediction accuracy on the test set"
    )
    parser.add_argument("--model-type",  default="monai",
                        choices=list(MODEL_REGISTRY),
                        help="Model architecture (default: monai)")
    parser.add_argument("--target-mode", default=DEFAULT_TARGET_MODE,
                        choices=list(TARGET_MODES),
                        help="Training target generation method the model "
                             f"was trained with (default: {DEFAULT_TARGET_MODE})")
    parser.add_argument("--model",       default=None,
                        help="Path to model weights "
                             "(default: data/models/<target-mode>/<model-type>/best.pt)")
    parser.add_argument("--data-root",   default="data/dataset")
    parser.add_argument("--results-dir", default=None,
                        help="Directory for results "
                             "(default: data/results/<target-mode>/<model-type>)")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Heatmap value threshold for peak detection (default: 0.5)")
    parser.add_argument("--nms-radius",  type=int,   default=20,
                        help="Min px distance between two peaks (default: 20)")
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--workers",     type=int,   default=4)
    parser.add_argument("--device",      default="",
                        help="torch device, e.g. 'cuda:0' (default: auto)")
    args = parser.parse_args()

    if args.model is None:
        args.model = default_model_path(args.model_type, args.target_mode)
    if args.results_dir is None:
        args.results_dir = default_results_dir(args.model_type, args.target_mode)

    evaluate(
        model_path=args.model,
        model_type=args.model_type,
        target_mode=args.target_mode,
        data_root=args.data_root,
        threshold=args.threshold,
        nms_radius=args.nms_radius,
        batch_size=args.batch_size,
        workers=args.workers,
        device_str=args.device,
        results_root=args.results_dir,
    )


if __name__ == "__main__":
    main()
