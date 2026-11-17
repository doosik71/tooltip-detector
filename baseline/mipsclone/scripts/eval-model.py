#!/usr/bin/env python3
"""Evaluate a trained RTMDet-Pose checkpoint on a dataset split.

Two families of numbers, as the other baselines report:

  detection   AP@0.5, AP@0.5:0.95, precision and recall of the `tool` boxes
  tip         miss rate, hit-rate @ 10/20/50 px, Hungarian TP/FP/FN and match
              distances, with the same matching rules as the root project's
              scripts/eval-model.py (common/tipmetrics.py is a verbatim copy of
              yolo26clone's)

Each tool box at or above `--conf` contributes the tip its pose branch
predicts; that set of points is what the tip metrics score.

`--boxes gt` replaces the detector with the ground-truth tool boxes and scores
the pose branch alone. It is an upper bound, not a comparable number: in this
dataset the tip lies on the edge of its box in ~97 % of tools, so a perfect
box already gives much of the answer away. Detection metrics are not computed
in that mode.

Everything is measured in original frame pixels (736 x 480).

Outputs, under baseline/mipsclone/data/results/<dataset>/<split>[-gtbox]/:

    summary.json   all metrics plus the run parameters
    per_tip.csv    one row per ground-truth tip

Usage:
    ./baseline/mipsclone/run eval-model --dataset cholec80 --device cuda:1
    ./baseline/mipsclone/run eval-model --dataset cholec80 --boxes gt
"""

import argparse
import csv
import json
import os

import numpy as np

if True:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from common.dataset import SPLITS, ToolPoseDataset, available_datasets, default_data_root
    from common.inference import (DEFAULT_CONF, DEFAULT_MAX_DET, DEFAULT_NMS_IOU, Predictor,
                                  default_model_path, results_dir)
    from common.metrics import DetectionEvaluator
    from common.progress import progress
    from common.tipmetrics import TipEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate RTMDet-Pose on a tooltip-detector dataset")
    parser.add_argument("--dataset", required=True, choices=available_datasets() or None)
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--model", default=None,
                        help="checkpoint (default: data/model/<dataset>/model.pt)")
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--device", default=None)
    parser.add_argument("--boxes", default="det", choices=("det", "gt"),
                        help="det: the model's own detections (default); gt: ground-truth "
                             "tool boxes, an upper bound for the pose branch")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"box confidence for the tip metrics (default: {DEFAULT_CONF})")
    parser.add_argument("--map-conf", type=float, default=0.001,
                        help="lower threshold used for the AP curves")
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    args.model = args.model or default_model_path(args.dataset)
    if not os.path.exists(args.model):
        raise SystemExit(f"checkpoint not found: {args.model}\n"
                         "train one first with scripts/train-model.py")
    predictor = Predictor(args.model, args.device)
    print(f"model   : {args.model}  [{predictor.device}]")
    print(f"trained : dataset={predictor.dataset} epoch={predictor.epoch} arch={predictor.arch}")

    dataset = ToolPoseDataset(args.dataset, args.split, predictor.image_size, augment=False,
                              data_root=args.data_root, frame_stride=args.frame_stride,
                              limit=args.limit)
    print(f"frames  : {len(dataset):,}  ({args.dataset}/{args.split}, boxes={args.boxes})")

    detection_eval = DetectionEvaluator(1, ("tool",))
    tip_eval = TipEvaluator()
    per_tip_rows: list[dict] = []
    total_ms, n_frames = 0.0, 0

    for index in progress(range(len(dataset)), desc="eval"):
        frame, labels = dataset.read_frame(index)
        gt_boxes = labels[:, :4]
        gt_tips = [tuple(row[4:6]) for row in labels if row[6] > 0]

        if args.boxes == "gt":
            predicted = predictor.tips_for_boxes(frame, gt_boxes)
            predicted_tips = [tuple(t) for t in predicted]
            tip_scores = [None] * len(predicted_tips)       # no detector score to report
        else:
            result, elapsed_ms = predictor.predict(frame, args.map_conf, args.max_det,
                                                   args.nms_iou, pose_conf=args.conf)
            total_ms += elapsed_ms
            detections = np.concatenate([result["boxes"], result["scores"][:, None],
                                         np.zeros((len(result["boxes"]), 1))], 1)
            ground_truth = np.concatenate([np.zeros((len(gt_boxes), 1)), gt_boxes], 1)
            detection_eval.add(detections, ground_truth)
            confident = result["scores"] >= args.conf
            predicted_tips = [tuple(t) for t in result["tips"][confident]]
            tip_scores = [float(s) for s in result["scores"][confident]]
        n_frames += 1

        assigned = tip_eval.add(gt_tips, predicted_tips)
        frame_name, session = dataset.frame_name(index), dataset.session_id(index)
        for gt_index, (gx, gy) in enumerate(gt_tips):
            hungarian = {f"hungarian_{int(cap)}px":
                         "assigned" if gt_index in matched else "unassigned"
                         for cap, matched in assigned.items()}
            row = {"frame": frame_name, "session": session,
                   "gt_x": round(float(gx), 2), "gt_y": round(float(gy), 2)}
            if predicted_tips:
                distances = [float(np.hypot(gx - px, gy - py)) for px, py in predicted_tips]
                nearest = int(np.argmin(distances))
                row.update({"pred_x": round(float(predicted_tips[nearest][0]), 2),
                            "pred_y": round(float(predicted_tips[nearest][1]), 2),
                            "score": "" if tip_scores[nearest] is None
                                     else round(tip_scores[nearest], 4),
                            "dist_px": round(distances[nearest], 2), "missed": 0})
            else:
                row.update({"pred_x": "", "pred_y": "", "score": "", "dist_px": "", "missed": 1})
            per_tip_rows.append({**row, **hungarian})

    tip_metrics = tip_eval.compute()
    detection_metrics = detection_eval.compute() if args.boxes == "det" else None

    split_dir = args.split if args.boxes == "det" else f"{args.split}-gtbox"
    output_dir = args.output_dir or os.path.join(results_dir(args.dataset), split_dir)
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "model": os.path.abspath(args.model),
        "trained_on": predictor.dataset,
        "trained_epoch": predictor.epoch,
        "arch": predictor.arch,
        "dataset": args.dataset,
        "split": args.split,
        "boxes": args.boxes,
        "n_frames": n_frames,
        "conf_threshold": args.conf,
        "map_conf_threshold": args.map_conf,
        "max_det": args.max_det,
        "nms_iou": args.nms_iou,
        "frame_stride": args.frame_stride,
        "image_size": predictor.image_size,
        "device": str(predictor.device),
        "ms_per_frame": round(total_ms / max(1, n_frames), 2) if args.boxes == "det" else None,
        "detection": detection_metrics,
        "tip": tip_metrics,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(output_dir, "per_tip.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "session", "gt_x", "gt_y", "pred_x",
                                                    "pred_y", "score", "dist_px", "missed"] +
                                [f"hungarian_{int(cap)}px" for cap in tip_eval.caps])
        writer.writeheader()
        writer.writerows(per_tip_rows)

    print()
    if detection_metrics is not None:
        tool = detection_metrics["per_class"]["tool"]
        print(f"tool       AP@0.5 {_fmt(tool['ap50'])}  AP@0.5:0.95 {_fmt(tool['ap50_95'])}  "
              f"P {_fmt(tool['precision'])}  R {_fmt(tool['recall'])}  (n_gt {tool['n_gt']:,})")
    print(f"tip        miss {tip_metrics['miss_rate_pct']:.2f} %   "
          f"hit@10 {tip_metrics['hit_rate_10px_pct']:.2f} %   "
          f"hit@20 {tip_metrics['hit_rate_20px_pct']:.2f} %   "
          f"hit@50 {tip_metrics['hit_rate_50px_pct']:.2f} %")
    print(f"           median {tip_metrics['median_dist_px']} px   "
          f"mean {tip_metrics['mean_dist_px']} px   p90 {tip_metrics['p90_dist_px']} px")
    for cap, row in tip_metrics["hungarian"].items():
        print(f"  hungarian {cap:>5s}  P {row['precision']:.4f}  R {row['recall']:.4f}  "
              f"(TP {row['tp']:,} FP {row['fp']:,} FN {row['fn']:,})")
    if summary["ms_per_frame"] is not None:
        print(f"speed      {summary['ms_per_frame']} ms/frame ({predictor.device})")
    print(f"written    {output_dir}")


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
