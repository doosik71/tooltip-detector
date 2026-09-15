"""Overlay drawing for the demo.

Predictions and ground truth are drawn in different visual languages so both
stay readable at once:

  prediction    solid box in the class colour, with a filled "tool 0.87" tag;
                a `tip` prediction also gets a cross at its centre, which is
                the coordinate the tip metrics are computed from
  ground truth  thin white box plus a white circle on the annotated tool tip
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_COLORS = {
    "tool": "#3CB44B",
    "tip": "#E6194B",
}
_FALLBACK_COLOR = "#AAAAAA"

_GT_COLOR = "#FFFFFF"
_GT_TIP_RADIUS = 7

_BOX_WIDTH = 3
_GT_BOX_WIDTH = 1
_TAG_PADDING = 3
_CROSS_ARM = 7


def class_color(label: str) -> str:
    return CLASS_COLORS.get(label, _FALLBACK_COLOR)


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:      # Pillow < 10.1: no size argument
        return ImageFont.load_default()


def draw_detections(frame_rgb: np.ndarray, detections: np.ndarray,
                    class_names: tuple[str, ...]) -> np.ndarray:
    """Draw (n, 6) [x1, y1, x2, y2, score, class] detections onto a copy."""
    pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil)
    font = _font()

    for x1, y1, x2, y2, score, class_id in detections:
        label = class_names[int(class_id)] if int(class_id) < len(class_names) else str(int(class_id))
        color = class_color(label)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=_BOX_WIDTH)

        if label == "tip":
            # The metrics use the box centre, so mark it explicitly -- a 10 px
            # box on its own is hard to read as a point.
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            draw.line([cx - _CROSS_ARM, cy, cx + _CROSS_ARM, cy], fill=color, width=2)
            draw.line([cx, cy - _CROSS_ARM, cx, cy + _CROSS_ARM], fill=color, width=2)

        text = f"{label} {score:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tag_w = right - left + 2 * _TAG_PADDING
        tag_h = bottom - top + 2 * _TAG_PADDING
        tag_y = y1 - tag_h if y1 - tag_h >= 0 else y1
        tag_x = min(x1, pil.width - tag_w)
        draw.rectangle([tag_x, tag_y, tag_x + tag_w, tag_y + tag_h], fill=color)
        draw.text((tag_x + _TAG_PADDING - left, tag_y + _TAG_PADDING - top),
                  text, fill="black", font=font)

    return np.array(pil)


def draw_ground_truth(frame_rgb: np.ndarray, annotations) -> np.ndarray:
    """Draw the dataset's annotated boxes and tips onto a copy."""
    pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil)

    for item in annotations:
        box = item.get("bbox")
        if box:
            x, y = box["x"], box["y"]
            draw.rectangle([x, y, x + box["width"], y + box["height"]],
                           outline=_GT_COLOR, width=_GT_BOX_WIDTH)
        tip = item.get("tip")
        if tip:
            r = _GT_TIP_RADIUS
            draw.ellipse([tip["x"] - r, tip["y"] - r, tip["x"] + r, tip["y"] + r],
                         outline=_GT_COLOR, width=2)

    return np.array(pil)
