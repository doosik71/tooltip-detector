"""Shared modules for the yolov8stooltip standalone model.

`scripts/*.py` add `standalone/yolov8stooltip` to sys.path and import from
here. The package is this project's own copy, started from the YOLOv8s clone
baseline (baseline/yolov8sclone) and changed where the pose head needs it;
nothing is imported from the baselines, and nothing imports `ultralytics`.

  model      YOLOv8s backbone and neck with a YOLOv8-pose head (tool + tip)
  boxes      box conversions, CIoU, letterbox, NMS carrying keypoints
  assigner   TaskAlignedAssigner, unchanged from the baseline
  loss       v8 detection loss plus the OKS-style keypoint loss
  dataset    data/dataset/<name> annotations -> one (box, tip) row per tool
  metrics    AP@0.5, AP@0.5:0.95, and tip boxes for a baseline-comparable tip AP
  progress   shared tqdm settings (ASCII bar, fixed width)
  tipmetrics hit-rate @ N px as in the root project, plus tool-tip pair hits
  inference  checkpoint format and single-frame inference
"""
