"""Shared modules for the YOLOv8s baseline.

`scripts/*.py` add `baseline/yolov8s` to sys.path and import from here, the
same convention the root project uses for its `ttd` package.

Training this repository's own 2-class (`tool` / `tip`) detector:

  dataset    data/dataset/<name> annotations -> a 2-class Ultralytics dataset,
             plus every path this baseline reads or writes
  inference  trained checkpoint + sidecar metadata, and single-frame detection
  metrics    AP@0.5, AP@0.5:0.95, precision, recall
  tipmetrics hit-rate @ N px, mirroring the root project's tip evaluation
  boxes      the IoU matrix the detection metrics are built on
  progress   shared tqdm settings (ASCII bar, fixed width)

Running the published 7-class Hugging Face checkpoint (the original demo):

  detector   loading yolov8s_cholec80.pt and running it on a single frame
  sources    discovering and reading frames from videos / extracted frame dirs
  draw       overlaying detections and ground-truth annotations on a frame

The two halves are deliberately separate: `detector.py` wraps a model that
classifies seven instrument types and was trained elsewhere, while
`inference.py` wraps the `tool`/`tip` model `scripts/train-model.py` produces
here. They answer different questions and are not interchangeable.

Training itself is Ultralytics'; nothing is reimplemented here. What lives in
this package is the glue that makes YOLOv8s read this repository's annotations
and report the same numbers as the other baselines.
"""
