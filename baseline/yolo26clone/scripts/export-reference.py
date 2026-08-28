#!/usr/bin/env python3
"""Extract the reference YOLO26 checkpoint into a plain tensor file.

This is the one script in this sub-project that needs `ultralytics`, and it is
the reason the rest does not: it turns the reference checkpoint -- an
Ultralytics pickle that cannot even be unpickled without the package -- into a
file of plain tensors that `scripts/verify-clone.py` can read anywhere.

Written out:

    state_dict     every parameter and buffer of the reference model
    input          one fixed random batch (seeded, so it is reproducible)
    one2many/*     that batch's raw head outputs, both branches
    one2one/*
    detections     the end-to-end output after top-k selection
    targets        a fixed set of ground-truth boxes
    loss_*         the reference's own loss on those targets, per branch

The clone is judged against all of them: identical keys and shapes proves the
architecture, identical outputs on the same input proves the forward pass, and
identical losses on the same targets proves the assigner and the loss -- the
parts that have no weights to compare and would otherwise go unchecked.

Run it with the sibling sub-project's virtualenv, which has ultralytics:

    uv run --project baseline/yolo26 python baseline/yolo26clone/scripts/export-reference.py

`./baseline/yolo26clone/run export-reference` will not work: this project's
environment deliberately has no ultralytics in it.
"""

import argparse
import os

import torch

if True:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from common.inference import data_dir

DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "yolo26", "data", "pretrained", "yolo26s.pt")
DEFAULT_OUTPUT = os.path.join(data_dir(), "reference", "yolo26s-reference.pt")

IMAGE_SIZE = 640
SEED = 0
N_TARGET_BOXES = 6


def random_targets(num_classes: int) -> torch.Tensor:
    """A fixed set of ground-truth boxes, in this project's target format.

    (n, 6) rows of [image index, class, cx, cy, w, h], normalised. Seeded from
    SEED + 1 so the boxes do not move when the input image does.
    """
    generator = torch.Generator().manual_seed(SEED + 1)
    centres = torch.rand(N_TARGET_BOXES, 2, generator=generator) * 0.6 + 0.2
    sizes = torch.rand(N_TARGET_BOXES, 2, generator=generator) * 0.3 + 0.05
    classes = torch.randint(0, num_classes, (N_TARGET_BOXES, 1), generator=generator).float()
    image_index = torch.zeros(N_TARGET_BOXES, 1)
    return torch.cat([image_index, classes, centres, sizes], dim=1)


def reference_losses(model, preds: dict, targets: torch.Tensor) -> dict:
    """The reference's loss on those targets: each branch alone, and blended.

    The two branches are scored separately as well as together, so a mismatch
    points at one assignment rather than at "the loss".
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import E2ELoss

    # A checkpoint carries its training arguments as a plain dict, and the loss
    # wants attribute access to the loss gains. Rebuilding them from the
    # package defaults also pins the gains the golden values were made with:
    # box 7.5, cls 0.5, dfl 1.5.
    model.args = get_cfg()

    batch = {"batch_idx": targets[:, 0], "cls": targets[:, 1], "bboxes": targets[:, 2:]}
    criterion = E2ELoss(model)
    # The reference keeps the three terms as a vector and the trainer sums it;
    # this project's loss returns that sum directly, so sum here to compare.
    results = {}
    for name in ("one2many", "one2one"):
        total, parts = getattr(criterion, name).loss(preds[name], batch)
        results[name] = (float(total.sum()), {k: float(v) for k, v in parts.items()})
    total, parts = criterion(preds, batch)
    results["blended"] = (float(total.sum()), {k: float(v) for k, v in parts.items()})
    gains = {"box": model.args.box, "cls": model.args.cls, "dfl": model.args.dfl,
             "o2m_weight": criterion.o2m}
    return results, gains


def main():
    parser = argparse.ArgumentParser(
        description="Export the reference YOLO26 checkpoint as plain tensors")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help=f"Ultralytics checkpoint (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"where the plain tensor file goes (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--scale", default="s", help="depth/width scale of the checkpoint")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "this script needs ultralytics, which this sub-project deliberately does not have.\n"
            "run it with the sibling project's environment:\n"
            "  uv run --project baseline/yolo26 python baseline/yolo26clone/scripts/export-reference.py")

    if not os.path.exists(args.weights):
        raise SystemExit(f"checkpoint not found: {args.weights}\n"
                         "download it first:  ./baseline/yolo26/run train-model --dataset cholec80 "
                         "(or any run that fetches the COCO weights)")

    model = YOLO(args.weights).model.float().eval()
    detect = model.model[-1]
    print(f"reference: {args.weights}")
    print(f"           nc={detect.nc} reg_max={detect.reg_max} end2end={detect.end2end} "
          f"stride={detect.stride.tolist()}")
    print(f"           {sum(p.numel() for p in model.parameters()):,} parameters")

    torch.manual_seed(SEED)
    image = torch.rand(1, 3, args.image_size, args.image_size)
    targets = random_targets(int(detect.nc))

    with torch.no_grad():
        detections, preds = model(image)

    losses, gains = reference_losses(model, preds, targets)
    print(f"gains    : box {gains['box']}  cls {gains['cls']}  dfl/l1 {gains['dfl']}  "
          f"one2many weight {gains['o2m_weight']}")
    for name, (total, parts) in losses.items():
        print(f"loss     : {name:9s} {total:.6f}  " +
              "  ".join(f"{k} {v:.6f}" for k, v in parts.items()))

    payload = {
        "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "nc": int(detect.nc),
        "reg_max": int(detect.reg_max),
        "scale": args.scale,
        "image_size": args.image_size,
        "seed": SEED,
        "input": image.cpu(),
        "detections": detections.cpu(),
        "targets": targets.cpu(),
        **{f"loss_{name}_total": total for name, (total, _) in losses.items()},
        **{f"loss_{name}_parts": parts for name, (_, parts) in losses.items()},
        "loss_gains": gains,
        **{f"{branch}_{key}": preds[branch][key].cpu()
           for branch in ("one2many", "one2one") for key in ("boxes", "scores")},
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(payload, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"written  : {args.output}  ({size_mb:.1f} MB)")
    print(f"next     : ./baseline/yolo26clone/run verify-clone")


if __name__ == "__main__":
    main()
