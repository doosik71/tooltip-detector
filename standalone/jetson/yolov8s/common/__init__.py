"""Shared modules for the Jetson yolov8s conversion/inference project.

`scripts/*.py` add `standalone/jetson/yolov8s` to sys.path and import from
here, the same convention every `baseline/*` sub-project uses for its own
`common` package.

  model      YOLOv8s architecture, anchor generation, DFL decoding
             (verbatim copy of baseline/yolov8sclone/common/model.py --
             see that file's docstring for why it must stay identical)
  boxes      box conversions, letterbox, NMS
             (verbatim copy of baseline/yolov8sclone/common/boxes.py)
  draw       prediction and ground-truth overlays
             (verbatim copy of baseline/yolov8sclone/common/draw.py)
  sources    video / frame-directory reading for the demo
             (copy of baseline/yolov8s/common/sources.py, own repo_root() depth)
  inference  TensorRT engine + model-info.json sidecar, single-frame inference
"""
