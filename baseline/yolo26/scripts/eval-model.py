#!/usr/bin/env python3
"""Evaluate a trained YOLO26 checkpoint on a dataset split.

Two families of numbers are reported, because this baseline is being compared
in two directions:

  detection   AP@0.5, AP@0.5:0.95, precision and recall for both classes
              (`tool`, `tip`) -- the usual object-detection metrics.
  tip         miss rate, hit-rate @ 10/20/50 px and match distances, computed
              from the centres of the predicted `tip` boxes with the same
              matching rules as the root project's scripts/eval-model.py, so
              the two are directly comparable.

Both are computed here rather than read from Ultralytics' own validator: the
CLAD-Net and YOLOv8s-clone baselines report these exact definitions, and a
number is only comparable if it was produced the same way. Frames and
annotations come straight from data/dataset/, not from the prepared YOLO
dataset, so evaluation does not depend on how training was set up -- only the
tip box size does, and that is read from the checkpoint's sidecar.

Everything is measured in original frame pixels (736 x 480), not in the
letterboxed 640 x 640 network input.

Outputs, under baseline/yolo26/data/results/<dataset>/<split>/ by default:

    summary.json   all metrics plus the run parameters
    per_tip.csv    one row per ground-truth tip (coordinates, nearest
                   prediction, distance, whether it was missed)

Usage:
    ./baseline/yolo26/run eval-model --dataset cholec80
    ./baseline/yolo26/run eval-model --dataset cholec80 --split val --limit 2000
"""

import argparse
import csv
import json
import os

import numpy as np

if True:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from common.dataset import (CLASS_NAMES, SPLITS, SplitFrames, available_datasets,
                                default_data_root, results_dir)
    from common.inference import (DEFAULT_CONF, DEFAULT_MAX_DET, Detector, class_index,
                                  default_model_path)
    from common.metrics import DetectionEvaluator
    from common.progress import progress
    from common.tipmetrics import TipEvaluator


def ground_truth_boxes(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised [class, cx, cy, w, h] -> pixel [class, x1, y1, x2, y2]."""
    if not len(labels):
        return np.zeros((0, 5), dtype=np.float32)
    cx, cy = labels[:, 1] * width, labels[:, 2] * height
    w, h = labels[:, 3] * width, labels[:, 4] * height
    return np.stack([labels[:, 0], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO26 on a tooltip-detector dataset")
    parser.add_argument("--dataset", required=True, choices=available_datasets() or None)
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--model", default=None,
                        help="checkpoint to evaluate (default: data/model/<dataset>/model.pt)")
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"confidence threshold (default: {DEFAULT_CONF})")
    parser.add_argument("--map-conf", type=float, default=0.001,
                        help="lower threshold used for the AP curves; the tip metrics "
                             "use --conf, since they need one decision per frame")
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET,
                        help="cap on boxes per frame. YOLO26 is end-to-end, so this "
                             "replaces the NMS IoU threshold the other baselines take")
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
    tip_class = class_index(detector.class_names, "tip")
    print(f"model   : {args.model}  [{detector.device}]")
    print(f"trained : dataset={detector.dataset} epoch={detector.epoch} "
          f"image_size={detector.image_size} tip_box={detector.tip_box_size:g} px"
          f"{'' if detector.has_info else '  (no model-info.json; defaults assumed)'}")
    if not detector.end2end:
        print("note    : this checkpoint is not an end-to-end head; --max-det still applies")

    # The tip box side is whatever the checkpoint was trained with; using a
    # different one here would score the model against labels it never saw.
    frames = SplitFrames(args.dataset, args.split, args.data_root,
                         frame_stride=args.frame_stride, limit=args.limit,
                         tip_box_size=detector.tip_box_size)
    print(f"frames  : {len(frames):,}  ({args.dataset}/{args.split})")

    detection_eval = DetectionEvaluator(len(detector.class_names), detector.class_names)
    tip_eval = TipEvaluator()
    per_tip_rows: list[dict] = []
    total_ms, n_frames = 0.0, 0

    for index in progress(range(len(frames)), desc="eval"):
        frame, labels = frames.read_frame(index)
        height, width = frame.shape[:2]
        session = frames.session_id(index)
        frame_name = frames.frame_name(index)

        # One forward pass at the lower threshold; the tip metrics then filter
        # the same detections at --conf, so the model runs once per frame.
        detections, elapsed_ms = detector.detect(frame, args.map_conf, args.max_det)
        total_ms += elapsed_ms
        n_frames += 1

        detection_eval.add(detections, ground_truth_boxes(labels, width, height))

        gt_tips = [((row[1] * width), (row[2] * height))
                   for row in labels if int(row[0]) == tip_class]
        confident = detections[detections[:, 4] >= args.conf] if len(detections) else detections
        tip_boxes = confident[confident[:, 5] == tip_class] if len(confident) else confident
        predicted_tips = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in tip_boxes]
        scores = [float(box[4]) for box in tip_boxes]

        tip_eval.add(gt_tips, predicted_tips)
        for gx, gy in gt_tips:
            if predicted_tips:
                distances = [float(np.hypot(gx - px, gy - py)) for px, py in predicted_tips]
                nearest = int(np.argmin(distances))
                per_tip_rows.append({"frame": frame_name, "session": session,
                                     "gt_x": round(gx, 2), "gt_y": round(gy, 2),
                                     "pred_x": round(predicted_tips[nearest][0], 2),
                                     "pred_y": round(predicted_tips[nearest][1], 2),
                                     "score": round(scores[nearest], 4),
                                     "dist_px": round(distances[nearest], 2), "missed": 0})
            else:
                per_tip_rows.append({"frame": frame_name, "session": session,
                                     "gt_x": round(gx, 2), "gt_y": round(gy, 2),
                                     "pred_x": "", "pred_y": "", "score": "",
                                     "dist_px": "", "missed": 1})

    detection_metrics = detection_eval.compute()
    tip_metrics = tip_eval.compute()

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
        "max_det": args.max_det,
        "end2end": detector.end2end,
        "frame_stride": args.frame_stride,
        "image_size": detector.image_size,
        "tip_box_size": detector.tip_box_size,
        "device": str(detector.device),
        "ms_per_frame": round(total_ms / max(1, n_frames), 2),
        "fps": round(1000.0 * n_frames / max(1e-9, total_ms), 1),
        "detection": detection_metrics,
        "tip": tip_metrics,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(os.path.join(output_dir, "per_tip.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "session", "gt_x", "gt_y",
                                                    "pred_x", "pred_y", "score",
                                                    "dist_px", "missed"])
        writer.writeheader()
        writer.writerows(per_tip_rows)

    per_class = detection_metrics["per_class"]
    print()
    print(f"detection  mAP@0.5 {_fmt(detection_metrics['map50'])}   "
          f"mAP@0.5:0.95 {_fmt(detection_metrics['map50_95'])}")
    for name in detector.class_names:
        row = per_class[name]
        print(f"  {name:5s} AP@0.5 {_fmt(row['ap50'])}  AP@0.5:0.95 {_fmt(row['ap50_95'])}  "
              f"P {_fmt(row['precision'])}  R {_fmt(row['recall'])}  (n_gt {row['n_gt']:,})")
    print(f"tip        miss {tip_metrics['miss_rate_pct']:.2f} %   "
          f"hit@10 {tip_metrics['hit_rate_10px_pct']:.2f} %   "
          f"hit@20 {tip_metrics['hit_rate_20px_pct']:.2f} %   "
          f"hit@50 {tip_metrics['hit_rate_50px_pct']:.2f} %")
    print(f"           median {tip_metrics['median_dist_px']} px   "
          f"mean {tip_metrics['mean_dist_px']} px   p90 {tip_metrics['p90_dist_px']} px")
    print(f"speed      {summary['ms_per_frame']} ms/frame ({summary['fps']} FPS, {detector.device})")
    print(f"written    {output_dir}")


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
