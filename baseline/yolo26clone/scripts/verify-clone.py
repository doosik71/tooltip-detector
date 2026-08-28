#!/usr/bin/env python3
"""Check this reimplementation against the reference YOLO26.

Four questions, in increasing order of strength:

  1. Does it have the right *shape*?      parameter counts per module
  2. Does it have the right *parameters*? every key and shape of the reference
                                          `state_dict`, loaded strictly
  3. Does it *compute* the right thing?   the reference's own outputs on a
                                          fixed input, reproduced to within
                                          floating-point noise
  4. Does it *learn* the right thing?     the reference's own loss on a fixed
                                          set of ground-truth boxes. This is
                                          the only check that reaches the
                                          assigner and the loss, which have no
                                          weights of their own to compare.

Question 1 is answered from this file alone. The rest need the reference
weights and outputs, extracted once by scripts/export-reference.py -- which is
the only script here that touches `ultralytics`, and is run with the sibling
sub-project's environment:

    uv run --project baseline/yolo26 python baseline/yolo26clone/scripts/export-reference.py
    ./baseline/yolo26clone/run verify-clone

Passing all four means the rebuild is not merely the right size: it is the
same function, trained by the same objective.

Usage:
    ./baseline/yolo26clone/run verify-clone
    ./baseline/yolo26clone/run verify-clone --scale s --nc 2      # shape only
"""

import argparse
import os

import torch

if True:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from common.inference import data_dir
    from common.loss import DetectionLoss
    from common.model import YOLO26, build, decode, parameter_count, postprocess

DEFAULT_REFERENCE = os.path.join(data_dir(), "reference", "yolo26s-reference.pt")

# How the module list groups into the parts the README talks about.
SECTIONS = (("0-10", "backbone (Conv, C3k2, SPPF, C2PSA)", range(0, 11)),
            ("13, 16", "top-down (C3k2)", (13, 16)),
            ("17, 19, 20, 22", "bottom-up (Conv, C3k2)", (17, 19, 20, 22)),
            ("23", "Detect (one2many + one2one branches)", (23,)))

# Anything bigger than this and the two are not the same computation; the
# reference runs in float32 and so does the clone, so the gap should be at the
# level of summation order, not of arithmetic.
TOLERANCE = 1e-4

# The losses are sums over thousands of anchors, so they accumulate more
# rounding than a single activation does. Still far below any real difference.
LOSS_TOLERANCE = 1e-3


def section_table(model: YOLO26) -> list[tuple[str, str, int]]:
    rows = []
    for label, description, indices in SECTIONS:
        total = sum(parameter_count(model.model[i]) for i in indices)
        rows.append((label, description, total))
    return rows


def compare_scalar(name: str, ours: float, theirs: float) -> bool:
    gap = abs(ours - theirs)
    ok = gap <= LOSS_TOLERANCE * max(1.0, abs(theirs))
    print(f"  {name:22s} {ours:12.6f} vs {theirs:12.6f}   |diff| = {gap:.3e}   "
          f"{'OK' if ok else 'MISMATCH'}")
    return ok


def compare(name: str, ours: torch.Tensor, theirs: torch.Tensor) -> bool:
    if ours.shape != theirs.shape:
        print(f"  {name:22s} SHAPE MISMATCH  {tuple(ours.shape)} vs {tuple(theirs.shape)}")
        return False
    gap = (ours - theirs).abs().max().item()
    ok = gap <= TOLERANCE
    print(f"  {name:22s} max |diff| = {gap:.3e}   {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify the YOLO26 clone against the reference implementation")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help=f"plain tensor file from export-reference.py "
                             f"(default: {DEFAULT_REFERENCE})")
    parser.add_argument("--scale", default=None, help="override the scale to build")
    parser.add_argument("--nc", type=int, default=None, help="override the class count")
    args = parser.parse_args()

    reference = None
    if os.path.exists(args.reference):
        reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    elif args.scale is None and args.nc is None:
        print(f"no reference file at {args.reference}")
        print("only the parameter table can be produced. To compare against the reference:")
        print("  uv run --project baseline/yolo26 python "
              "baseline/yolo26clone/scripts/export-reference.py")
        print()

    scale = args.scale or (reference or {}).get("scale", "s")
    num_classes = args.nc or (reference or {}).get("nc", 2)

    model = build(num_classes=num_classes, scale=scale).eval()
    print(f"clone   : YOLO26{scale}, nc={num_classes}, "
          f"{parameter_count(model):,} parameters")
    print()
    print(f"{'modules':<16} {'contents':<40} {'parameters':>12}")
    for label, description, count in section_table(model):
        print(f"{label:<16} {description:<40} {count:>12,}")
    print(f"{'total':<16} {'':<40} {parameter_count(model):>12,}")

    if reference is None:
        return

    print()
    print(f"reference: {args.reference}  (nc={reference['nc']}, scale={reference['scale']})")
    if (num_classes, scale) != (reference["nc"], reference["scale"]):
        # The reference is a COCO checkpoint; a model of another size or class
        # count simply has different weights, so there is nothing to compare.
        print(f"           built nc={num_classes} scale={scale} instead, so only the "
              "parameter table above applies")
        print("           drop --nc/--scale to compare against the reference")
        return

    # 2. the parameters themselves
    missing, unexpected = model.load_state_dict(reference["state_dict"], strict=False)
    ours = set(model.state_dict())
    theirs = set(reference["state_dict"])
    print()
    print("state_dict")
    print(f"  keys           clone {len(ours):,}   reference {len(theirs):,}   "
          f"missing {len(missing)}   unexpected {len(unexpected)}")
    for key in list(missing)[:5]:
        print(f"    missing:    {key}")
    for key in list(unexpected)[:5]:
        print(f"    unexpected: {key}")
    keys_ok = not missing and not unexpected

    # 3. the computation
    print()
    print(f"forward pass on the reference's own input {tuple(reference['input'].shape)}")
    with torch.no_grad():
        preds = model(reference["input"])
        detections = postprocess(decode(preds["one2one"]), model.detect.max_det)

    outputs_ok = all([
        compare("one2many boxes", preds["one2many"]["boxes"], reference["one2many_boxes"]),
        compare("one2many scores", preds["one2many"]["scores"], reference["one2many_scores"]),
        compare("one2one boxes", preds["one2one"]["boxes"], reference["one2one_boxes"]),
        compare("one2one scores", preds["one2one"]["scores"], reference["one2one_scores"]),
        compare("detections", detections, reference["detections"]),
    ])

    losses_ok = True
    if "loss_blended_total" in reference:
        print()
        print(f"loss on the reference's own {len(reference['targets'])} ground-truth boxes")
        criterion = DetectionLoss(model, epochs=1)
        branches = {"one2many": criterion.one2many, "one2one": criterion.one2one}
        checks = []
        for name, branch in branches.items():
            total, parts = branch(preds[name], reference["targets"])
            checks.append(compare_scalar(f"{name} total", float(total),
                                         reference[f"loss_{name}_total"]))
            for key, value in parts.items():
                # The reference names the three terms box_loss/cls_loss/l1_loss.
                checks.append(compare_scalar(f"{name} {key}", value,
                                             reference[f"loss_{name}_parts"][f"{key}_loss"]))
        total, _ = criterion(preds, reference["targets"])
        checks.append(compare_scalar("blended total", float(total),
                                     reference["loss_blended_total"]))
        losses_ok = all(checks)

    print()
    if keys_ok and outputs_ok and losses_ok:
        print("PASS: same parameters, same outputs, same loss. "
              "The reimplementation matches.")
    else:
        print("FAIL: see the mismatches above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
