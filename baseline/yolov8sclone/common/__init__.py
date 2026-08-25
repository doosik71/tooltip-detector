"""Shared modules for the YOLOv8s clone baseline.

`scripts/*.py` add `baseline/yolov8sclone` to sys.path and import from here,
the same convention the root project uses for its `ttd` package. Nothing in
this package imports `ultralytics`; only what the root project already depends
on (torch, torchvision, numpy, opencv, pillow, scipy, tqdm) is used.

  model      YOLOv8s architecture, anchor generation, DFL decoding
  boxes      box conversions, CIoU, letterbox, NMS
  assigner   TaskAlignedAssigner -- YOLOv8's dynamic label assignment
  loss       BCE(cls) + CIoU(box) + DFL, the v8 detection loss
  dataset    data/dataset/<name> annotations -> 2-class detection labels
  metrics    AP@0.5, AP@0.5:0.95, recall
  progress   shared tqdm settings (ASCII bar, fixed width)
  tipmetrics hit-rate @ N px, mirroring the root project's tip evaluation
  inference  checkpoint format and single-frame inference
  sources    video / frame-directory reading for the demo
  draw       prediction and ground-truth overlays
"""
