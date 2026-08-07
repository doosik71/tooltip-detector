#!/usr/bin/env python3
"""Compare test-set inference speed between tooltip-detector models.

Runs the same test split through each selected model and writes a compact speed
comparison report to ``data/results/<dataset>/<target-mode>/speed-comparison.json``
by default.
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ttd.checkpoints import default_model_path
from ttd.dataset import (DATASETS, DEFAULT_TARGET_MODE, TARGET_MODES, SurgicalToolDataset,
                         require_samples)
from ttd.model import REGISTRY as MODEL_REGISTRY
from ttd.model import build as build_model
from ttd.transforms import _eval_transform


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _build_sample_indices(dataset_size: int, num_samples: int, seed: int) -> list[int]:
    if dataset_size <= 0:
        return []

    gen = torch.Generator().manual_seed(seed)
    count = min(num_samples, dataset_size)
    indices = torch.randperm(dataset_size, generator=gen)[:count].tolist()
    indices.sort()
    return indices


def _build_loader(
    data_root: str,
    target_mode: str,
    sample_indices: list[int],
    batch_size: int,
    workers: int,
    device: torch.device,
):
    full_ds = SurgicalToolDataset(data_root, "test", transform=_eval_transform(),
                                    target_mode=target_mode)
    ds = Subset(full_ds, sample_indices) if sample_indices else Subset(full_ds, [])
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    return full_ds, ds, loader


def _warmup_model(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    warmup_batch_size: int,
) -> None:
    if len(dataset) == 0:
        return

    image, _ = dataset[0]
    batch = image.unsqueeze(0).repeat(warmup_batch_size, 1, 1, 1)
    batch = batch.to(device, dtype=torch.float32)

    with torch.inference_mode():
        _ = model(batch)
    _synchronize(device)


def benchmark_model(
    model_type: str,
    model_path: str,
    dataset_name: str,
    target_mode: str,
    data_root: str,
    sample_indices: list[int],
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict:
    model = build_model(model_type, num_classes=2).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    full_dataset, dataset, loader = _build_loader(
        data_root,
        target_mode,
        sample_indices,
        batch_size,
        workers,
        device,
    )
    measured_batch_size = min(batch_size, len(dataset)) if len(dataset) else 1
    _warmup_model(model, dataset, device, measured_batch_size)

    n_frames = len(dataset)
    n_batches = len(loader)
    forward_time_sec = 0.0

    wall_t0 = time.perf_counter()
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, dtype=torch.float32, non_blocking=device.type == "cuda")
            _synchronize(device)
            infer_t0 = time.perf_counter()
            _ = model(images)
            _synchronize(device)
            forward_time_sec += time.perf_counter() - infer_t0
    wall_time_sec = time.perf_counter() - wall_t0

    forward_ms = forward_time_sec * 1000.0
    wall_ms = wall_time_sec * 1000.0

    return {
        "model_type": model_type,
        "dataset": dataset_name,
        "target_mode": target_mode,
        "model_path": model_path,
        "parameter_count": _count_parameters(model),
        "n_test_frames_total": len(full_dataset),
        "n_test_frames_sampled": n_frames,
        "n_batches": n_batches,
        "batch_size": batch_size,
        "forward_time_total_ms": round(forward_ms, 2),
        "forward_time_per_frame_ms": round(forward_ms / max(1, n_frames), 4),
        "forward_time_per_batch_ms": round(forward_ms / max(1, n_batches), 4),
        "forward_fps": round(n_frames / max(forward_time_sec, 1e-12), 2),
        "wall_time_total_ms": round(wall_ms, 2),
        "wall_time_per_frame_ms": round(wall_ms / max(1, n_frames), 4),
        "wall_fps": round(n_frames / max(wall_time_sec, 1e-12), 2),
    }


def _build_comparison(results: list[dict]) -> dict:
    by_type = {item["model_type"]: item for item in results}
    comparison: dict[str, float | str] = {}

    if "monai" in by_type and "monai_mini" in by_type:
        monai = by_type["monai"]
        mini = by_type["monai_mini"]
        comparison = {
            "faster_model_by_forward_time": (
                "monai_mini"
                if mini["forward_time_per_frame_ms"] < monai["forward_time_per_frame_ms"]
                else "monai"
            ),
            "faster_model_by_wall_time": (
                "monai_mini"
                if mini["wall_time_per_frame_ms"] < monai["wall_time_per_frame_ms"]
                else "monai"
            ),
            "monai_vs_monai_mini_forward_speedup": round(
                monai["forward_time_per_frame_ms"] / max(mini["forward_time_per_frame_ms"], 1e-12),
                4,
            ),
            "monai_vs_monai_mini_wall_speedup": round(
                monai["wall_time_per_frame_ms"] / max(mini["wall_time_per_frame_ms"], 1e-12),
                4,
            ),
        }

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare inference speed of tooltip-detector models on the test split."
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=["monai", "monai_mini"],
        choices=list(MODEL_REGISTRY),
        help="Model types to benchmark (default: monai monai_mini)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASETS),
        help="Dataset name under --data-root (e.g. cholec80)",
    )
    parser.add_argument(
        "--target-mode",
        default=DEFAULT_TARGET_MODE,
        choices=list(TARGET_MODES),
        help=f"Which trained checkpoint variant to load (default: {DEFAULT_TARGET_MODE})",
    )
    parser.add_argument("--data-root", default="data/dataset",
                        help="Root directory containing <dataset>/ subdirectories "
                             "(default: data/dataset)")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="",
        help="Torch device string (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: data/results/<dataset>/<target-mode>/speed-comparison.json)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            "data", "results", args.dataset, args.target_mode, "speed-comparison.json")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dataset_root = os.path.join(args.data_root, args.dataset)

    print(f"Device      : {device}")
    print(f"Dataset      : {args.dataset}")
    print(f"Data root    : {dataset_root}")
    print(f"Target mode  : {args.target_mode}")
    print(f"Samples      : {args.num_samples}")
    print(f"Seed         : {args.seed}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Workers      : {args.workers}")
    print(f"Output       : {args.output}")
    print()

    base_dataset = SurgicalToolDataset(dataset_root, "test", transform=_eval_transform(),
                                        target_mode=args.target_mode)
    # Checked here, before any model is loaded: _build_loader() rebuilds the
    # same dataset per model, so one check up front covers every benchmark run.
    require_samples(base_dataset, "test", dataset_root)
    sample_indices = _build_sample_indices(len(base_dataset), args.num_samples, args.seed)

    results: list[dict] = []
    for model_type in args.model_types:
        model_path = default_model_path(
            model_type=model_type, dataset_name=args.dataset,
            target_mode=args.target_mode)
        print(f"[Benchmark] {model_type}")
        print(f"  model: {model_path}")
        result = benchmark_model(
            model_type=model_type,
            model_path=model_path,
            dataset_name=args.dataset,
            target_mode=args.target_mode,
            data_root=dataset_root,
            sample_indices=sample_indices,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
        )
        results.append(result)
        print(
            f"  forward/frame={result['forward_time_per_frame_ms']:.4f} ms"
            f"  wall/frame={result['wall_time_per_frame_ms']:.4f} ms"
            f"  forward_fps={result['forward_fps']:.2f}"
            f"  wall_fps={result['wall_fps']:.2f}"
        )
        print()

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "dataset": args.dataset,
        "data_root": dataset_root,
        "target_mode": args.target_mode,
        "split": "test",
        "num_samples_requested": args.num_samples,
        "num_samples_used": len(sample_indices),
        "sample_seed": args.seed,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "metrics_note": {
            "forward_time": "Model forward pass only, excluding data loading and post-processing.",
            "wall_time": "End-to-end loop time over the test DataLoader, including host-side overhead.",
        },
        "results": results,
        "comparison": _build_comparison(results),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved speed comparison to {args.output}")


if __name__ == "__main__":
    main()
