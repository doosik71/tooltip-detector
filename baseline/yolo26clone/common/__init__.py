"""Shared modules for the YOLO26 clone baseline.

`scripts/*.py` add `baseline/yolo26clone` to sys.path and import from here,
the same convention the root project uses for its `ttd` package. Nothing in
this package imports `ultralytics`; only what the root project already depends
on (torch, torchvision, numpy, opencv, pillow, scipy, tqdm) is used.

  model      YOLO26 architecture, anchor generation, end-to-end decoding
  assigner   TaskAlignedAssigner, with YOLO26's box floor and second top-k
  loss       BCE(cls) + CIoU(box) + L1, over both head branches
  optim      MuSGD, the hybrid Muon/SGD optimizer YOLO26 trains with
  boxes      box conversions, CIoU, letterbox
  dataset    data/dataset/<name> annotations -> 2-class detection labels
  metrics    AP@0.5, AP@0.5:0.95, recall
  progress   shared tqdm settings (ASCII bar, fixed width)
  tipmetrics hit-rate @ N px, mirroring the root project's tip evaluation
  inference  checkpoint format and single-frame inference
  sources    video / frame-directory reading for the demo
  draw       prediction and ground-truth overlays

The one exception is scripts/export-reference.py, which needs `ultralytics` to
read the reference checkpoint once. It is not imported from anywhere here.
"""
