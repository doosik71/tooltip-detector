"""Shared modules for the CLAD-Net baseline.

`scripts/*.py` add `baseline/cladnet` to sys.path and import from here, the
same convention the root project uses for its `ttd` package.

  modules    the paper's building blocks: AAB, MSAB, CAM, RM
  backbone   CSPDarknet53 (YOLOv5-style) feature extractor
  neck       the Cross-Layer Aggregated Attention Module
  model      backbone + neck + decoupled head, plus decode/NMS
  boxes      box conversions, CIoU, NMS, letterbox mapping
  dataset    data/dataset/<name> annotations -> 2-class detection labels
  loss       YOLOv5-style label assignment and the paper's loss
  metrics    AP@0.5, AP@0.5:0.95, recall
  progress   shared tqdm settings (ASCII bar, fixed width)
  tipmetrics hit-rate @ N px, mirroring the root project's tip evaluation
"""
