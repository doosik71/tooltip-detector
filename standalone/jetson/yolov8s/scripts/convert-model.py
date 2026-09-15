#!/usr/bin/env python3
"""Convert a yolov8sclone `model.pt` checkpoint into a Jetson TensorRT engine.

The checkpoint (`baseline/yolov8sclone/data/model/<dataset>/<label-set>/model.pt`)
is a plain-PyTorch YOLOv8s clone -- see `common/model.py`. This script:

  1. rebuilds the model from the checkpoint's own `arch`/`class_names` and
     loads its weights (identical to `yolov8sclone.common.inference.Detector`);
  2. wraps the forward pass together with `common.model.decode` into a single
     graph, so the exported model's output is already `(1, N, 4 + nc)` boxes
     and class scores -- the same tensor shape
     `yolov8sclone.common.boxes.non_max_suppression` consumes today;
  3. exports that graph to ONNX at a fixed `image_size x image_size` input;
  4. calls `trtexec` to build a TensorRT engine from the ONNX model.

NMS is deliberately *not* part of the exported graph: it produces a
variable-length output (however many boxes survive thresholding), which does
not fit a fixed-shape TensorRT engine. It runs in Python instead, exactly as
it does today in `yolov8sclone.common.boxes.non_max_suppression` --
`scripts/demo.py` calls it after the engine returns its raw predictions.

Why a TensorRT engine must be built on this Jetson itself, not elsewhere:
TensorRT engines are optimized for one specific GPU and TensorRT build, and
refuse to load on any other -- see standalone/jetson/yolov8s/README.md.

Usage
-----
    ./standalone/jetson/yolov8s/run convert-model
    ./standalone/jetson/yolov8s/run convert-model --precision fp32
    ./standalone/jetson/yolov8s/run convert-model \\
        --checkpoint ../../../baseline/yolov8sclone/data/model/erop/tiponly/model.pt
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn

# NOTE: deliberately not a top-level import. `common.*` needs this project on
# sys.path, which only exists after the insert() below runs. Nesting the
# imports under `if True:` keeps an editor's "organize imports" from hoisting
# them back above the sys.path fix-up. Same convention as every baseline's
# scripts/*.py.
if True:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from common.model import build, decode, fuse

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", "..", ".."))

DEFAULT_CHECKPOINT = os.path.join(
    REPO_ROOT, "baseline", "yolov8sclone", "data", "model", "cholec80", "tooltip", "model.pt")
DEFAULT_TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

ONNX_FILENAME = "model.onnx"
ENGINE_FILENAME = "model.engine"
INFO_FILENAME = "model-info.json"


class DecodedModel(nn.Module):
    """`model(x)` followed by `decode(...)` as one graph.

    `decode` needs the `YOLOv8` instance itself (it reads `.detect` and
    `.strides` off it), not just its output, so it cannot be exported on its
    own -- this wrapper is what makes it part of the traced graph.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return decode(self.model(x), self.model)


def load_model(checkpoint_path: str):
    """Rebuild and load a checkpoint, fused for inference.

    Mirrors `yolov8sclone.common.inference.Detector.__init__` exactly: the
    engine has to match that code's numerics, since `fuse` folds BatchNorm
    into the preceding convolution and changes the weights the model computes
    with (see `common/model.py`'s `fuse` docstring).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = checkpoint.get("arch", {})
    class_names = tuple(checkpoint.get("class_names", ()))
    model = build(num_classes=len(class_names), **arch)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    fuse(model)
    return model, checkpoint, class_names


def export_onnx(model: nn.Module, image_size: int, opset: int, onnx_path: str) -> None:
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    wrapped = DecodedModel(model)
    wrapped.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (dummy,),
            onnx_path,
            input_names=["images"],
            output_names=["predictions"],
            opset_version=opset,
            dynamic_axes=None,          # fixed batch=1, image_size x image_size
            do_constant_folding=True,
        )


def build_engine(trtexec: str, onnx_path: str, engine_path: str,
                 precision: str, workspace_mb: int | None) -> None:
    if not os.path.exists(trtexec):
        raise FileNotFoundError(
            f"trtexec not found: {trtexec}\n"
            "It ships with JetPack's TensorRT apt packages "
            "(libnvinfer-bin) -- this script cannot substitute for it.")
    command = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if precision == "fp16":
        command.append("--fp16")
    if workspace_mb:
        command.append(f"--memPoolSize=workspace:{workspace_mb}M")
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a yolov8sclone model.pt checkpoint to a Jetson TensorRT engine")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help=f"yolov8sclone checkpoint to convert (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--output-dir", default=None,
                        help="where model.onnx/model.engine/model-info.json are written "
                             "(default: data/model/<dataset>/<label_set>/, read from the "
                             "checkpoint's own metadata)")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16",
                        help="TensorRT engine precision (default: fp16, see README.md for why)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--workspace-mb", type=int, default=None,
                        help="trtexec builder workspace size in MiB (default: trtexec's own default)")
    parser.add_argument("--trtexec", default=DEFAULT_TRTEXEC,
                        help=f"path to the trtexec binary (default: {DEFAULT_TRTEXEC})")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if model.engine already exists")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, checkpoint, class_names = load_model(args.checkpoint)
    image_size = int(checkpoint.get("image_size", 640))
    dataset = checkpoint.get("dataset", "unknown")
    label_set = checkpoint.get("label_set", "tooltip")
    print(f"  dataset={dataset}  label_set={label_set}  classes={class_names}  "
          f"image_size={image_size}  epoch={checkpoint.get('epoch')}")

    output_dir = args.output_dir or os.path.join(
        PROJECT_DIR, "data", "model", dataset, label_set)
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, ONNX_FILENAME)
    engine_path = os.path.join(output_dir, ENGINE_FILENAME)
    info_path = os.path.join(output_dir, INFO_FILENAME)

    if os.path.exists(engine_path) and not args.force:
        print(f"Already converted: {engine_path} (use --force to rebuild)")
        return

    print(f"Exporting ONNX (opset {args.opset}, {image_size}x{image_size}): {onnx_path}")
    export_onnx(model, image_size, args.opset, onnx_path)

    print(f"Building TensorRT engine ({args.precision}): {engine_path}")
    started = time.perf_counter()
    build_engine(args.trtexec, onnx_path, engine_path, args.precision, args.workspace_mb)
    elapsed_s = time.perf_counter() - started
    print(f"Engine built in {elapsed_s:.1f}s")

    info = {
        "source_checkpoint": os.path.relpath(args.checkpoint, output_dir),
        "class_names": list(class_names),
        "label_set": label_set,
        "dataset": dataset,
        "image_size": image_size,
        "tip_box_size": checkpoint.get("tip_box_size"),
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
        "precision": args.precision,
        "opset": args.opset,
    }
    with open(info_path, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    print(f"Wrote {info_path}")


if __name__ == "__main__":
    main()
