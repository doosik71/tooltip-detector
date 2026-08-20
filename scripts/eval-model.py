"""Evaluate legacy and one-to-one tooltip peak matching metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ttd.checkpoints import default_model_path, default_results_dir
from ttd.dataset import DATASETS, DEFAULT_TARGET_MODE, TARGET_MODES, SurgicalToolDataset, require_samples
from ttd.evaluation import hungarian_match
from ttd.model import REGISTRY as MODEL_REGISTRY
from ttd.model import build as build_model
from ttd.peaks import find_peaks
from ttd.transforms import _eval_transform

_HIT_THRESHOLDS = (10, 20, 50)
_DEFAULT_MATCH_DISTANCES = (10.0, 20.0, 50.0)


def _session_id(ann_path: str) -> str:
    return os.path.basename(ann_path).rsplit("_", 1)[0]


def _new_legacy_accumulator() -> dict:
    return {"distances": [], "n_gt": 0, "n_missed": 0, "hits": Counter(),
            "sessions": defaultdict(lambda: {"distances": [], "n_gt": 0, "n_missed": 0})}


def _new_hungarian_accumulator(caps: tuple[float, ...]) -> dict:
    return {cap: {"distances": [], "tp": 0, "fp": 0, "fn": 0,
                  "sessions": defaultdict(lambda: {"distances": [], "tp": 0, "fp": 0, "fn": 0})}
            for cap in caps}


def _accumulate_legacy(acc: dict, session: str, gt: list[tuple[float, float]], candidates: list[tuple[int, int, float]]):
    acc["n_gt"] += len(gt)
    acc["sessions"][session]["n_gt"] += len(gt)
    if not candidates:
        acc["n_missed"] += len(gt)
        acc["sessions"][session]["n_missed"] += len(gt)
        return [(None, None)] * len(gt), ["unassigned"] * len(gt)
    matches = []
    for gx, gy in gt:
        index = min(range(len(candidates)), key=lambda i: np.hypot(gx - candidates[i][0], gy - candidates[i][1]))
        distance = float(np.hypot(gx - candidates[index][0], gy - candidates[index][1]))
        matches.append((index, distance))
        acc["distances"].append(distance)
        acc["sessions"][session]["distances"].append(distance)
        for threshold in _HIT_THRESHOLDS:
            if distance <= threshold:
                acc["hits"][threshold] += 1
    uses = Counter(index for index, _ in matches)
    return matches, ["shared_legacy" if uses[index] > 1 else "assigned" for index, _ in matches]


def _accumulate_hungarian(acc: dict, session: str, gt: list[tuple[float, float]], candidates: list[tuple[int, int, float]]):
    by_cap = {}
    for cap, values in acc.items():
        assignments = {match.gt_index: (match.prediction_index, match.distance_px)
                       for match in hungarian_match(gt, candidates, cap)}
        by_cap[cap] = assignments
        tp, fn, fp = len(assignments), len(gt) - len(assignments), len(candidates) - len(assignments)
        values["tp"] += tp
        values["fn"] += fn
        values["fp"] += fp
        values["distances"].extend(distance for _, distance in assignments.values())
        session_values = values["sessions"][session]
        session_values["tp"] += tp
        session_values["fn"] += fn
        session_values["fp"] += fp
        session_values["distances"].extend(distance for _, distance in assignments.values())
    return by_cap


def _legacy_stats(acc: dict, metadata: dict) -> dict:
    distances = np.asarray(acc["distances"], dtype=float)
    safe_n = max(1, acc["n_gt"])
    result = {**metadata, "n_gt_tips": acc["n_gt"], "n_missed": acc["n_missed"],
              "miss_rate_pct": round(acc["n_missed"] / safe_n * 100, 2),
              "mean_dist_px": round(float(distances.mean()), 2) if distances.size else None,
              "median_dist_px": round(float(np.median(distances)), 2) if distances.size else None,
              "p90_dist_px": round(float(np.percentile(distances, 90)), 2) if distances.size else None,
              **{f"hit_rate_{threshold}px_pct": round(acc["hits"][threshold] / safe_n * 100, 2)
                 for threshold in _HIT_THRESHOLDS}, "per_session": {}}
    for session, values in sorted(acc["sessions"].items()):
        session_distances = np.asarray(values["distances"], dtype=float)
        result["per_session"][session] = {"n_gt_tips": values["n_gt"], "n_missed": values["n_missed"],
            "mean_dist_px": round(float(session_distances.mean()), 2) if session_distances.size else None}
    return result


def _hungarian_stats(acc: dict) -> dict:
    output = {}
    for cap, values in acc.items():
        distances = np.asarray(values["distances"], dtype=float)
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        key = str(int(cap) if cap.is_integer() else cap)
        output[key] = {"max_distance_px": cap, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(2 * precision * recall / max(1e-12, precision + recall), 6),
            "mean_dist_px": round(float(distances.mean()), 2) if distances.size else None,
            "median_dist_px": round(float(np.median(distances)), 2) if distances.size else None,
            "p90_dist_px": round(float(np.percentile(distances, 90)), 2) if distances.size else None,
            "per_session": {}}
        for session, session_values in sorted(values["sessions"].items()):
            session_distances = np.asarray(session_values["distances"], dtype=float)
            stp, sfp, sfn = session_values["tp"], session_values["fp"], session_values["fn"]
            sp, sr = stp / max(1, stp + sfp), stp / max(1, stp + sfn)
            output[key]["per_session"][session] = {"tp": stp, "fp": sfp, "fn": sfn,
                "precision": round(sp, 6), "recall": round(sr, 6),
                "f1": round(2 * sp * sr / max(1e-12, sp + sr), 6),
                "mean_dist_px": round(float(session_distances.mean()), 2) if session_distances.size else None}
    return output


def _build_model(model_path: str, model_type: str, device: torch.device):
    model = build_model(model_type, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    return model


def _dataset_loader(data_root: str, split: str, target_mode: str, batch_size: int, workers: int, device: torch.device):
    dataset = SurgicalToolDataset(data_root, split, transform=_eval_transform(), target_mode=target_mode)
    require_samples(dataset, split, data_root)
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                               pin_memory=device.type == "cuda")


def estimate_bias(model_path, model_type, dataset_name, target_mode, data_root, threshold, nms_radius,
                  peak_method, batch_size, workers, device_str, near_distance_px):
    if near_distance_px <= 0:
        raise ValueError("near_distance_px must be positive")
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset, loader = _dataset_loader(data_root, "val", target_mode, batch_size, workers, device)
    model = _build_model(model_path, model_type, device)
    deltas, by_session = [], defaultdict(list)
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(loader):
            heatmaps = torch.sigmoid(model(images.to(device, dtype=torch.float32))[:, 1]).cpu().numpy()
            for offset in range(len(images)):
                ann_path = dataset.samples[batch_index * batch_size + offset]
                with open(ann_path, encoding="utf-8") as handle:
                    gt = [(item["tip"]["x"], item["tip"]["y"]) for item in json.load(handle)["annotations"]]
                candidates = find_peaks(heatmaps[offset], threshold, nms_radius, peak_method)
                for gx, gy in gt:
                    if candidates:
                        px, py, _ = min(candidates, key=lambda point: np.hypot(gx - point[0], gy - point[1]))
                        if np.hypot(gx - px, gy - py) <= near_distance_px:
                            delta = (float(px - gx), float(py - gy))
                            deltas.append(delta)
                            by_session[_session_id(ann_path)].append(delta)
    if not deltas:
        raise RuntimeError("No validation matches found within the bias distance cap")
    dx, dy = np.median(np.asarray(deltas), axis=0)
    return {"dataset": dataset_name, "target_mode": target_mode, "model_type": model_type,
            "model_path": model_path, "split": "val", "threshold": threshold, "nms_radius": nms_radius, "peak_method": peak_method,
            "near_distance_px": near_distance_px, "n_matches": len(deltas),
            "dx_px": round(float(dx), 4), "dy_px": round(float(dy), 4),
            "per_session": {session: {"n_matches": len(items),
                "dx_px": round(float(np.median(np.asarray(items)[:, 0])), 4),
                "dy_px": round(float(np.median(np.asarray(items)[:, 1])), 4)}
                for session, items in sorted(by_session.items())}}


def evaluate(model_path, model_type, dataset_name, target_mode, data_root, threshold, nms_radius,
             peak_method, batch_size, workers, device_str, results_root, match_distances, bias=None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset, loader = _dataset_loader(data_root, "test", target_mode, batch_size, workers, device)
    model = _build_model(model_path, model_type, device)
    legacy, hungarian = _new_legacy_accumulator(), _new_hungarian_accumulator(match_distances)
    corrected_legacy = _new_legacy_accumulator() if bias else None
    corrected_hungarian = _new_hungarian_accumulator(match_distances) if bias else None
    rows = []
    print(f"Device: {device}; test frames: {len(dataset):,}; match caps: {match_distances}")
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(loader):
            heatmaps = torch.sigmoid(model(images.to(device, dtype=torch.float32))[:, 1]).cpu().numpy()
            for offset in range(len(images)):
                ann_path = dataset.samples[batch_index * batch_size + offset]
                stem, session = Path(ann_path).stem, _session_id(ann_path)
                with open(ann_path, encoding="utf-8") as handle:
                    gt = [(item["tip"]["x"], item["tip"]["y"]) for item in json.load(handle)["annotations"]]
                if not gt:
                    continue
                candidates = find_peaks(heatmaps[offset], threshold, nms_radius, peak_method)
                legacy_matches, legacy_types = _accumulate_legacy(legacy, session, gt, candidates)
                assignments = _accumulate_hungarian(hungarian, session, gt, candidates)
                corrected_candidates = None
                if bias:
                    corrected_candidates = [(x - bias["dx_px"], y - bias["dy_px"], score) for x, y, score in candidates]
                    corrected_matches, corrected_types = _accumulate_legacy(corrected_legacy, session, gt, corrected_candidates)
                    corrected_assignments = _accumulate_hungarian(corrected_hungarian, session, gt, corrected_candidates)
                for gt_index, ((gx, gy), (prediction_index, distance), legacy_type) in enumerate(zip(gt, legacy_matches, legacy_types, strict=True)):
                    row = {"frame": stem, "session": session, "gt_x": gx, "gt_y": gy,
                           "pred_x": "", "pred_y": "", "dist_px": "", "missed": int(prediction_index is None),
                           "match_type": legacy_type}
                    if prediction_index is not None:
                        px, py, _ = candidates[prediction_index]
                        row.update({"pred_x": px, "pred_y": py, "dist_px": round(distance, 2)})
                    for cap in match_distances:
                        key = str(int(cap) if cap.is_integer() else cap)
                        row[f"hungarian_{key}px"] = "assigned" if gt_index in assignments[cap] else "unassigned"
                    if bias:
                        corrected_index, corrected_distance = corrected_matches[gt_index]
                        row.update({"bias_pred_x": "", "bias_pred_y": "", "bias_dist_px": "",
                            "bias_missed": int(corrected_index is None), "bias_match_type": corrected_types[gt_index]})
                        if corrected_index is not None:
                            px, py, _ = corrected_candidates[corrected_index]
                            row.update({"bias_pred_x": round(px, 2), "bias_pred_y": round(py, 2),
                                        "bias_dist_px": round(corrected_distance, 2)})
                        for cap in match_distances:
                            key = str(int(cap) if cap.is_integer() else cap)
                            row[f"bias_hungarian_{key}px"] = "assigned" if gt_index in corrected_assignments[cap] else "unassigned"
                    rows.append(row)
            done = min((batch_index + 1) * batch_size, len(dataset))
            if done % 5000 < batch_size or done == len(dataset):
                print(f"  {done:>6}/{len(dataset)}")
    metadata = {"timestamp": time.strftime("%Y%m%d_%H%M%S"), "dataset": dataset_name,
        "model_type": model_type, "target_mode": target_mode, "model_path": model_path,
        "threshold": threshold, "nms_radius": nms_radius, "peak_method": peak_method, "data_root": data_root,
        "n_frames_with_tools": len({row["frame"] for row in rows})}
    stats = _legacy_stats(legacy, metadata)
    stats["hungarian"] = {"algorithm": "scipy.optimize.linear_sum_assignment",
                          "distance_caps_px": _hungarian_stats(hungarian)}
    if bias:
        corrected = _legacy_stats(corrected_legacy, metadata)
        corrected["hungarian"] = {"algorithm": "scipy.optimize.linear_sum_assignment",
                                   "distance_caps_px": _hungarian_stats(corrected_hungarian)}
        stats["bias_correction"] = {"applied": True, "bias": bias, "corrected": corrected}
    _save_results(stats, rows, results_root)
    _print_results(stats)
    return stats


def _save_results(stats, rows, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(results_dir, "per_tip.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["frame", "session"])
        writer.writeheader()
        writer.writerows(rows)


def _print_results(stats):
    print(f"Legacy: miss={stats['miss_rate_pct']:.2f}% median={stats['median_dist_px']} px")
    for cap, values in stats["hungarian"]["distance_caps_px"].items():
        print(f"Hungarian @{cap}px: TP={values['tp']:,} FP={values['fp']:,} FN={values['fn']:,} F1={values['f1']:.4f}")
    if "bias_correction" in stats:
        corrected = stats["bias_correction"]["corrected"]
        print(f"Bias-corrected legacy: miss={corrected['miss_rate_pct']:.2f}% median={corrected['median_dist_px']} px")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TooltipDetector tip predictions")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--model-type", default="monai", choices=list(MODEL_REGISTRY))
    parser.add_argument("--target-mode", default=DEFAULT_TARGET_MODE, choices=list(TARGET_MODES))
    parser.add_argument("--model", default=None)
    parser.add_argument("--data-root", default="data/dataset")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nms-radius", type=int, default=20)
    parser.add_argument("--peak-method", choices=("connected-components", "watershed"),
                        default="connected-components")
    parser.add_argument("--match-distance", type=float, nargs="+", default=list(_DEFAULT_MATCH_DISTANCES),
                        help="Hungarian assignment caps in px (default: 10 20 50)")
    parser.add_argument("--estimate-bias", action="store_true", help="Estimate and save val-set bias.json")
    parser.add_argument("--bias-distance", type=float, default=20.0)
    parser.add_argument("--apply-bias", action="store_true", help="Apply bias.json and report raw plus corrected test metrics")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    if any(value <= 0 for value in args.match_distance):
        parser.error("--match-distance values must be positive")
    model_path = args.model or default_model_path(args.model_type, args.dataset, args.target_mode)
    data_root = os.path.join(args.data_root, args.dataset)
    if args.estimate_bias:
        bias = estimate_bias(model_path, args.model_type, args.dataset, args.target_mode, data_root,
                             args.threshold, args.nms_radius, args.peak_method, args.batch_size, args.workers,
                             args.device, args.bias_distance)
        bias_path = Path(model_path).with_name("bias.json")
        bias_path.write_text(json.dumps(bias, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved validation bias to {bias_path}: dx={bias['dx_px']:+.2f}, dy={bias['dy_px']:+.2f} px")
        return
    bias = None
    if args.apply_bias:
        bias_path = Path(model_path).with_name("bias.json")
        if not bias_path.is_file():
            parser.error(f"--apply-bias requires {bias_path}; run --estimate-bias first")
        bias = json.loads(bias_path.read_text(encoding="utf-8"))
    evaluate(model_path, args.model_type, args.dataset, args.target_mode, data_root, args.threshold,
             args.nms_radius, args.peak_method, args.batch_size, args.workers, args.device,
             args.results_dir or default_results_dir(args.model_type, args.dataset, args.target_mode),
             tuple(sorted(set(args.match_distance))), bias)


if __name__ == "__main__":
    main()
