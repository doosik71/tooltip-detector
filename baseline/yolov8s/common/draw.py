"""Overlay drawing for the baseline demo.

Predictions and ground truth are drawn in deliberately different visual
languages so they stay readable when both are on screen at once:

  prediction   solid box in the class colour, with a filled "Class 0.87" tag
  ground truth thin white box plus a white circle on the annotated tool tip

The tooltip-detector ground truth has no class labels -- only a bounding box
and a tip coordinate per tool -- so there is nothing class-coloured to draw for
it, which is convenient: white reads as "reference" against the colours.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# One colour per class, for both checkpoints this baseline can show: the two
# classes scripts/train-model.py produces, and the seven of the published
# instrument classifier. Unknown ids fall back to _FALLBACK_COLOR, so an
# extended checkpoint still draws.
CLASS_COLORS = {
    # trained here: tool box and the 32 px box on the tool tip
    "tool": "#3CB44B",
    "tip": "#E6194B",
    # the published 7-class Hugging Face checkpoint
    "Bag": "#F58231",
    "Bipolar": "#4363D8",
    "Clipper": "#911EB4",
    "Grasper": "#3CB44B",
    "Hook": "#E6194B",
    "Irrigator": "#42D4F4",
    "Scissors": "#FFE119",
}
_FALLBACK_COLOR = "#AAAAAA"

_GT_COLOR = "#FFFFFF"
_GT_TIP_RADIUS = 6

_BOX_WIDTH = 3
_GT_BOX_WIDTH = 1
_TAG_PADDING = 3


def class_color(label: str) -> str:
    return CLASS_COLORS.get(label, _FALLBACK_COLOR)


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:      # Pillow < 10.1: no size argument
        return ImageFont.load_default()


def draw_detections(frame_rgb: np.ndarray, detections) -> np.ndarray:
    """Return a copy of *frame_rgb* with every Detection drawn on it."""
    pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil)
    font = _font()

    for det in detections:
        color = class_color(det.label)
        draw.rectangle([det.x1, det.y1, det.x2, det.y2],
                       outline=color, width=_BOX_WIDTH)

        text = f"{det.label} {det.conf:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tag_w = right - left + 2 * _TAG_PADDING
        tag_h = bottom - top + 2 * _TAG_PADDING

        # Sit the tag above the box, or just inside its top edge when the box
        # is already touching the top of the frame.
        tag_y = det.y1 - tag_h if det.y1 - tag_h >= 0 else det.y1
        tag_x = min(det.x1, pil.width - tag_w)
        draw.rectangle([tag_x, tag_y, tag_x + tag_w, tag_y + tag_h], fill=color)
        draw.text((tag_x + _TAG_PADDING - left, tag_y + _TAG_PADDING - top),
                  text, fill="black", font=font)

    return np.array(pil)


def draw_ground_truth(frame_rgb: np.ndarray, annotations) -> np.ndarray:
    """Return a copy of *frame_rgb* with the annotated boxes and tips drawn."""
    pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil)

    for ann in annotations:
        box = ann.get("bbox")
        if box:
            x, y = box["x"], box["y"]
            draw.rectangle([x, y, x + box["width"], y + box["height"]],
                           outline=_GT_COLOR, width=_GT_BOX_WIDTH)
        tip = ann.get("tip")
        if tip:
            r = _GT_TIP_RADIUS
            draw.ellipse([tip["x"] - r, tip["y"] - r, tip["x"] + r, tip["y"] + r],
                         outline=_GT_COLOR, width=2)

    return np.array(pil)
