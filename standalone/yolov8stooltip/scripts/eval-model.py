#!/usr/bin/env python3
"""Evaluate a trained yolov8stooltip checkpoint on a dataset split.

Three families of numbers are reported:

  detection   AP@0.5, AP@0.5:0.95, precision and recall for `tool`, and for a
              `tip` box of `--tip-box-size` drawn around each predicted tip --
              the form the baselines and the project's tip AP target use, so
              the numbers mean the same thing at the same box side.
  tip         miss rate, hit-rate @ 10/20/50 px and match distances over the
              predicted tips, with the same matching rules as the root
              project's scripts/eval-model.py.
  pairs       does each ground-truth tool get a predicted tool *and its own*
              tip (see common/tipmetrics.PairEvaluator)? The tip rules above
              cannot see a tip that went to the wrong tool; this can.

Everything is measured in original frame pixels (736 x 480), not in the
letterboxed 640 x 640 network input.

Outputs, under standalone/yolov8stooltip/data/results/<dataset>/<split>/ by default:

    summary.json   all metrics plus the run parameters
    per_tip.csv    one row per ground-truth tip, with the same columns as the
                   root project's and the baselines' per_tip.csv

Usage:
    ./standalone/yolov8stooltip/run eval-model --dataset cholec80 --device cuda:1
    ./standalone/yolov8stooltip/run eval-model --dataset cholec80 --split val --limit 2000
"""

import argparse
import csv
import json
import os

import numpy as np

if True:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from common.dataset import (DEFAULT_TIP_BOX_SIZE, SPLITS, SurgicalPoseDataset,
                                available_datasets, default_data_root)
    from common.inference import (DEFAULT_CONF, DEFAULT_IOU, Detector, default_model_path,
                                  results_dir)
    from common.metrics import DetectionEvaluator, ground_truth_with_tip_boxes, with_tip_boxes
    from common.progress import progress
    from common.tipmetrics import PairEvaluator, TipEvaluator

EVAL_NAMES = ("tool", "tip")


def ground_truth_tools(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised [class, cx, cy, w, h, kx, ky, kv] -> pixel [x1, y1, x2, y2, kx, ky]."""
    if not len(labels):
        return np.zeros((0, 6), dtype=np.float32)
    cx, cy = labels[:, 1] * width, labels[:, 2] * height
    w, h = labels[:, 3] * width, labels[:, 4] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
                     labels[:, 5] * width, labels[:, 6] * height], 1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate yolov8stooltip on a tooltip-detector dataset")
    parser.add_argument("--dataset", required=True, choices=available_datasets() or None)
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--model", default=None,
                        help="checkpoint to evaluate (default: data/model/<dataset>/model.pt)")
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"confidence threshold (default: {DEFAULT_CONF})")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU,
                        help=f"NMS IoU threshold (default: {DEFAULT_IOU})")
    parser.add_argument("--map-conf", type=float, default=0.001,
                        help="lower threshold used for the AP curves; the tip and pair "
                             "metrics use --conf, since they need one decision per frame")
    parser.add_argument("--tip-box-size", type=float, default=DEFAULT_TIP_BOX_SIZE,
                        help="side of the box drawn around each tip for the tip AP, in "
                             f"original-frame px (default: {DEFAULT_TIP_BOX_SIZE:g})")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="evaluate every Nth frame of the split")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None,
                        help="where summary.json goes (default: data/results/<dataset>/<split>)")
    args = parser.parse_args()

    args.model = args.model or default_model_path(args.dataset)
    if not os.path.exists(args.model):
        raise SystemExit(f"checkpoint not found: {args.model}\n"
                         "train one first with scripts/train-model.py")

    detector = Detector(args.model, args.device)
    print(f"model   : {args.model}  [{detector.device}]")
    print(f"trained : dataset={detector.dataset} epoch={detector.epoch} "
          f"image_size={detector.image_size} kpt_shape={detector.kpt_shape}")

    dataset = SurgicalPoseDataset(args.dataset, args.split, detector.image_size,
                                  augment=False, data_root=args.data_root,
                                  frame_stride=args.frame_stride, limit=args.limit)
    print(f"frames  : {len(dataset):,}  ({args.dataset}/{args.split})")

    detection_eval = DetectionEvaluator(len(EVAL_NAMES), EVAL_NAMES)
    tip_eval = TipEvaluator()
    pair_eval = PairEvaluator()
    per_tip_rows: list[dict] = []
    total_ms, n_frames = 0.0, 0

    for index in progress(range(len(dataset)), desc="eval"):
        frame, labels = dataset.read_frame(index)
        height, width = frame.shape[:2]
        session = dataset.session_id(index)
        frame_name = dataset.frame_name(index)

        # One forward pass at the lower threshold; the tip and pair metrics then
        # filter the same detections at --conf, so the model runs once per frame.
        detections, elapsed_ms = detector.detect(frame, args.map_conf, args.iou)
        total_ms += elapsed_ms
        n_frames += 1

        tools = ground_truth_tools(labels, width, height)
        detection_eval.add(with_tip_boxes(detections, args.tip_box_size),
                           ground_truth_with_tip_boxes(tools, args.tip_box_size))

        confident = detections[detections[:, 4] >= args.conf] if len(detections) else detections
        pair_eval.add(tools[:, :4], tools[:, 4:], confident[:, :4], confident[:, 6:8])

        gt_tips = [(float(x), float(y)) for x, y in tools[:, 4:]]
        predicted_tips = [(float(x), float(y)) for x, y in confident[:, 6:8]]
        scores = [float(score) for score in confident[:, 4]]
        assigned = tip_eval.add(gt_tips, predicted_tips)
        for gt_index, (gx, gy) in enumerate(gt_tips):
            # Same column names and values as the root project's per_tip.csv,
            # so success rates by tool count can be recomputed from this file.
            hungarian = {f"hungarian_{int(cap)}px":
                         "assigned" if gt_index in matched else "unassigned"
                         for cap, matched in assigned.items()}
            if predicted_tips:
                distances = [float(np.hypot(gx - px, gy - py)) for px, py in predicted_tips]
                nearest = int(np.argmin(distances))
                per_tip_rows.append({"frame": frame_name, "session": session,
                                     "gt_x": round(gx, 2), "gt_y": round(gy, 2),
                                     "pred_x": round(predicted_tips[nearest][0], 2),
                                     "pred_y": round(predicted_tips[nearest][1], 2),
                                     "score": round(scores[nearest], 4),
                                     "dist_px": round(distances[nearest], 2), "missed": 0,
                                     **hungarian})
            else:
                per_tip_rows.append({"frame": frame_name, "session": session,
                                     "gt_x": round(gx, 2), "gt_y": round(gy, 2),
                                     "pred_x": "", "pred_y": "", "score": "",
                                     "dist_px": "", "missed": 1, **hungarian})

    detection_metrics = detection_eval.compute()
    tip_metrics = tip_eval.compute()
    pair_metrics = pair_eval.compute()

    output_dir = args.output_dir or os.path.join(results_dir(args.dataset), args.split)
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "model": os.path.abspath(args.model),
        "trained_on": detector.dataset,
        "trained_epoch": detector.epoch,
        "dataset": args.dataset,
        "split": args.split,
        "n_frames": n_frames,
        "conf_threshold": args.conf,
        "map_conf_threshold": args.map_conf,
        "iou_threshold": args.iou,
        "frame_stride": args.frame_stride,
        "image_size": detector.image_size,
        "tip_box_size": args.tip_box_size,
        "kpt_shape": list(detector.kpt_shape),
        "device": str(detector.device),
        "ms_per_frame": round(total_ms / max(1, n_frames), 2),
        "fps": round(1000.0 * n_frames / max(1e-9, total_ms), 1),
        "detection": detection_metrics,
        "tip": tip_metrics,
        "pairs": pair_metrics,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(os.path.join(output_dir, "per_tip.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "session", "gt_x", "gt_y",
                                                    "pred_x", "pred_y", "score",
                                                    "dist_px", "missed"] +
                                [f"hungarian_{int(cap)}px" for cap in tip_eval.caps])
        writer.writeheader()
        writer.writerows(per_tip_rows)

    per_class = detection_metrics["per_class"]
    print()
    print(f"detection  mAP@0.5 {_fmt(detection_metrics['map50'])}   "
          f"mAP@0.5:0.95 {_fmt(detection_metrics['map50_95'])}   "
          f"(tip box {args.tip_box_size:g} px)")
    for name in EVAL_NAMES:
        row = per_class[name]
        print(f"  {name:5s} AP@0.5 {_fmt(row['ap50'])}  AP@0.5:0.95 {_fmt(row['ap50_95'])}  "
              f"P {_fmt(row['precision'])}  R {_fmt(row['recall'])}  (n_gt {row['n_gt']:,})")
    print(f"tip        miss {tip_metrics['miss_rate_pct']:.2f} %   "
          f"hit@10 {tip_metrics['hit_rate_10px_pct']:.2f} %   "
          f"hit@20 {tip_metrics['hit_rate_20px_pct']:.2f} %   "
          f"hit@50 {tip_metrics['hit_rate_50px_pct']:.2f} %")
    print(f"           median {tip_metrics['median_dist_px']} px   "
          f"mean {tip_metrics['mean_dist_px']} px   p90 {tip_metrics['p90_dist_px']} px")
    print(f"pairs      box matched {pair_metrics['box_matched_pct']:.2f} %   "
          f"own tip @10 {pair_metrics['pair_hit_rate_10px_pct']:.2f} %   "
          f"@20 {pair_metrics['pair_hit_rate_20px_pct']:.2f} %   "
          f"@50 {pair_metrics['pair_hit_rate_50px_pct']:.2f} %")
    print(f"speed      {summary['ms_per_frame']} ms/frame ({summary['fps']} FPS, {detector.device})")
    print(f"written    {output_dir}")


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
