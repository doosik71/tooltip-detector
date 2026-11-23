#!/usr/bin/env python3
"""Evaluate detrtooltip, or yolov8stooltip for reference, on the same frames.

By default the frames are the held-out checkpoint-selection videos (video42)
and the labels are resolved exactly as in training (manual, then opus, then
annotation/; see common/dataset.py). `--split` evaluates a whole split
instead; frames used in training then score optimistically.

Metrics (common/evaluation.py): tool AP, tip hit-rate and Hungarian
precision/recall over all tips, tool-tip pair hits for one or two tips, and
second-tip precision/recall on trusted labels.

`--family yolov8stooltip` runs a standalone/yolov8stooltip checkpoint through
that project's own `Detector` instead, so the two models are scored on the
same frames with the same rules. It never predicts a second tip.

Output: data/results/<dataset>/<name>/summary.json, where <name> is
`--name` or the model family and frame selection.

Usage:
    ./standalone/detrtooltip/run eval-model --dataset cholec80 --device cuda:1
    ./standalone/detrtooltip/run eval-model --dataset cholec80 --device cuda:1 \\
        --family yolov8stooltip
"""

import argparse
import importlib.util
import json
import os
import sys

import torch

if True:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from common.dataset import (TRUSTED_SOURCES, available_datasets, default_data_root,
                                list_frames, parse_label, read_rgb, repo_root)
    from common.evaluation import Evaluation
    from common.inference import Detector, model_dir, results_dir
    from common.progress import progress

# AP needs the low-scoring detections too; the tip metrics then apply --conf.
AP_CONF = 0.001


def default_model(dataset: str) -> str:
    return os.path.join(model_dir(dataset), "model.pt")


def yolov8stooltip_predictor(weights: str, device: str):
    """standalone/yolov8stooltip's Detector, as a frame -> annotations function."""
    project = os.path.join(repo_root(), "standalone", "yolov8stooltip")
    package_dir = os.path.join(project, "common")
    spec = importlib.util.spec_from_file_location(
        "yolov8stooltip_common", os.path.join(package_dir, "__init__.py"),
        submodule_search_locations=[package_dir])
    package = importlib.util.module_from_spec(spec)
    sys.modules["yolov8stooltip_common"] = package
    spec.loader.exec_module(package)
    inference = importlib.import_module("yolov8stooltip_common.inference")
    detector = inference.Detector(weights, device=device)

    def predict(frame_rgb):
        rows, _ = detector.detect(frame_rgb, conf=AP_CONF)
        return [{"bbox": {"x": float(r[0]), "y": float(r[1]), "width": float(r[2] - r[0]),
                          "height": float(r[3] - r[1])},
                 "tip": {"x": float(r[6]), "y": float(r[7])}, "score": float(r[4])}
                for r in rows]
    return predict


def main():
    parser = argparse.ArgumentParser(description="Evaluate detrtooltip (or yolov8stooltip)")
    parser.add_argument("--dataset", required=True, choices=available_datasets() or None)
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--family", default="detrtooltip", choices=("detrtooltip", "yolov8stooltip"))
    parser.add_argument("--model", default=None,
                        help="checkpoint (default: data/model/<dataset>/model.pt; "
                             "for yolov8stooltip its data/model/<dataset>/model.pt)")
    parser.add_argument("--videos", default="video42",
                        help="comma-separated videos to evaluate (ignored with --split)")
    parser.add_argument("--split", default=None, choices=("train", "val", "test"))
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.5,
                        help="score threshold for the tip and pair metrics")
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None, help="results sub-directory name")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.family == "detrtooltip":
        args.model = args.model or default_model(args.dataset)
        detector = Detector(args.model, device)
        predict = lambda frame: detector.predict(frame, conf=0.0)      # noqa: E731
    else:
        args.model = args.model or os.path.join(repo_root(), "standalone", "yolov8stooltip",
                                                "data", "model", args.dataset, "model.pt")
        predict = yolov8stooltip_predictor(args.model, device)
    if not os.path.exists(args.model):
        raise SystemExit(f"model not found: {args.model}")

    if args.split:
        frames = list_frames(args.dataset, args.data_root, splits=(args.split,),
                             frame_stride=args.frame_stride)
        selection = args.split
    else:
        videos = {v.strip() for v in args.videos.split(",") if v.strip()}
        frames = list_frames(args.dataset, args.data_root, videos=videos,
                             frame_stride=args.frame_stride)
        selection = "-".join(sorted(videos))
    frames = [f for f in frames if f.label is not None]
    if not frames:
        raise SystemExit("no labelled frames selected")
    print(f"{args.family}: {args.model}\n{len(frames):,} frames ({selection})")

    evaluation = Evaluation(args.conf)
    with torch.no_grad():
        for frame in progress(frames, desc="eval"):
            evaluation.add(parse_label(frame.label), predict(read_rgb(frame.image)),
                           frame.source in TRUSTED_SOURCES)
    result = evaluation.compute()
    result.update({"family": args.family, "model": os.path.abspath(args.model),
                   "selection": selection, "dataset": args.dataset})

    name = args.name or f"{args.family}-{selection}"
    output_dir = os.path.join(results_dir(args.dataset), name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    pairs, tips = result["pairs"], result["tips"]
    hung = tips["hungarian"]["20px"]
    print(f"tool AP50 {result['tool_ap50']}  AP50:95 {result['tool_ap50_95']}")
    print(f"tips: hit@10/20/50 {tips['hit_rate_10px_pct']} / {tips['hit_rate_20px_pct']} / "
          f"{tips['hit_rate_50px_pct']} %  Hungarian@20 P {hung['precision']} R {hung['recall']}  "
          f"F1 {result['selection_tip_f1_20px']}")
    print(f"pairs: box matched {pairs['box_matched_pct']} %  own tip @10/20 "
          f"{pairs['pair_hit_rate_10px_pct']} / {pairs['pair_hit_rate_20px_pct']} %  "
          f"two-tip tools @20 {pairs['two_tip_pair_hit_rate_20px_pct']} %")
    print(f"tip2: P {pairs['tip2']['precision']} R {pairs['tip2']['recall']} "
          f"F1 {pairs['tip2']['f1']}")
    print(f"-> {output_dir}/summary.json")


if __name__ == "__main__":
    main()
