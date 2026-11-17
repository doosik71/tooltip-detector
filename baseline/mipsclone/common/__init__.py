"""Shared modules for the RTMDet-Pose baseline (mipsclone).

`scripts/*.py` add `baseline/mipsclone` to sys.path and import from here, the
convention the other clone baselines use. Only packages the root project
already depends on (torch, torchvision, numpy, opencv, scipy, tqdm) are used;
nothing from OpenMMLab is imported.

  model      CSPNeXt + PAFPN + RTMDet head + RoI pose head (RTMPose's SimCC/GAU)
  loss       DynamicSoftLabelAssigner, QFL + GIoU, SimCC KL on jittered GT RoIs
  boxes      IoU / GIoU, distance decoding, letterbox
  dataset    data/dataset/<name> annotations -> tool boxes paired with tips
  metrics    AP@0.5, AP@0.5:0.95 (copied from yolo26clone)
  tipmetrics hit-rate @ N px and Hungarian matching (copied from yolo26clone)
  progress   shared tqdm settings
  inference  checkpoint format and single-frame inference
"""
