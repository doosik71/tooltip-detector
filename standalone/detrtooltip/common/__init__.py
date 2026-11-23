"""Shared modules for the detrtooltip standalone model.

`scripts/*.py` add `standalone/detrtooltip` to sys.path and import from here.
Nothing is imported from the other projects of the repository.

  model      ResNet + FPN backbone, Deformable DETR head, tip refinement,
             the stage-1 Pretrainer, and postprocessing
  msdeform   multi-scale deformable attention in plain PyTorch
  loss       masked reconstruction loss; Hungarian matching and the set loss
             with order-free tips and the second-tip existence term
  boxes      box conversions and IoU
  dataset    frames, label source priority, sampling weights, augmentation
  inference  checkpoint format and single-frame inference
  metrics    AP@0.5, AP@0.5:0.95 (copied from yolov8stooltip)
  tipmetrics hit-rate @ N px (copied) and tool-tip pair hits for 1-2 tips
  progress   shared tqdm settings (copied)
"""
