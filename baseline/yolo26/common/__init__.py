"""Shared modules for the YOLO26 baseline.

`scripts/*.py` add `baseline/yolo26` to sys.path and import from here, the same
convention the root project uses for its `ttd` package.

  dataset    data/dataset/<name> annotations -> a 2-class Ultralytics dataset,
             plus every path this baseline reads or writes
  inference  checkpoint + sidecar metadata, and single-frame detection
  metrics    AP@0.5, AP@0.5:0.95, precision, recall
  tipmetrics hit-rate @ N px, mirroring the root project's tip evaluation
  boxes      the IoU matrix the detection metrics are built on
  progress   shared tqdm settings (ASCII bar, fixed width)
  sources    video / frame-directory reading for the demo
  draw       prediction and ground-truth overlays

Training itself is Ultralytics'; nothing is reimplemented here. What lives in
this package is the glue that makes YOLO26 read this repository's annotations
and report the same numbers as the other baselines.
"""
