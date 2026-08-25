"""Shared modules for the YOLOv8s surgical-instrument-detection baseline.

`scripts/*.py` add `baseline/yolov8s` to sys.path and import from here, the
same convention the root project uses for its `ttd` package.

  detector : loading yolov8s_cholec80.pt and running it on a single frame
  sources  : discovering and reading frames from videos / extracted frame dirs
  draw     : overlaying detections and ground-truth annotations on a frame
"""
