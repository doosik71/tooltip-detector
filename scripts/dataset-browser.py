#!/usr/bin/env python3
"""Interactive dataset browser for SurgicalToolDataset.

Shows the original image alongside the distance-based heatmap overlay.
Navigate with ← → arrow keys or buttons.

Usage:
    uv run python scripts/dataset-browser.py [--data-root PATH] [--split SPLIT]
"""
import argparse
import json
import os
import sys

import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ttd.dataset import SurgicalToolDataset


SPLITS = ["train", "val", "test"]
PANEL_W = 552   # 736 × 0.75
PANEL_H = 360   # 480 × 0.75

# Colors cycled per tool index
_TOOL_COLORS = ["#00FF00", "#FF8800", "#00AAFF", "#FF00FF", "#FFFF00",
                "#FF4444", "#44FFFF", "#FF44FF"]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _colorize(target: np.ndarray) -> np.ndarray:
    """Float32 [0,1] → RGB uint8 using a hot colormap.

    Gradient: 0 → black, 1/3 → red, 2/3 → yellow, 1 → white.
    """
    r = np.clip(target * 3.0,       0.0, 1.0)
    g = np.clip(target * 3.0 - 1.0, 0.0, 1.0)
    b = np.clip(target * 3.0 - 2.0, 0.0, 1.0)
    return (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)


def _blend_heatmap(image: np.ndarray, target: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """Overlay colorized heatmap on the image where target > 0."""
    colored = _colorize(target)
    mask = (target > 0)[..., np.newaxis]
    return np.where(mask,
                    (image * (1.0 - alpha) + colored * alpha).astype(np.uint8),
                    image)


def _draw_annotations(image: np.ndarray, annotations: list) -> np.ndarray:
    """Draw bounding boxes and tip cross-hairs onto the image."""
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    for i, ann in enumerate(annotations):
        color = _TOOL_COLORS[i % len(_TOOL_COLORS)]
        b = ann["bbox"]
        draw.rectangle(
            [b["x"], b["y"], b["x"] + b["width"] - 1, b["y"] + b["height"] - 1],
            outline=color, width=2,
        )
        tx, ty = ann["tip"]["x"], ann["tip"]["y"]
        r = 5
        draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=color, outline="white")
        draw.line([tx - 12, ty, tx + 12, ty], fill="white", width=1)
        draw.line([tx, ty - 12, tx, ty + 12], fill="white", width=1)
    return np.array(pil)


def _to_photo(arr: np.ndarray) -> ImageTk.PhotoImage:
    pil = Image.fromarray(arr).resize((PANEL_W, PANEL_H), Image.BILINEAR)
    return ImageTk.PhotoImage(pil)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class DatasetBrowser(tk.Tk):
    def __init__(self, data_root: str, split: str):
        super().__init__()
        self.title("Dataset Browser — SurgicalToolDataset")
        self.resizable(False, False)

        self._data_root = data_root
        self._ds: SurgicalToolDataset | None = None
        self._idx = 0
        # Hold references so GC does not delete PhotoImages
        self._photos: list[ImageTk.PhotoImage] = []

        self._build_ui()
        self._load_split(split)

        self.bind("<Left>",  lambda _: self._navigate(-1))
        self.bind("<Right>", lambda _: self._navigate(+1))
        self.bind("<r>",     lambda _: self._random())

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        # ── Controls row ─────────────────────────────────────────────────
        ctrl = tk.Frame(self, pady=6, padx=8)
        ctrl.pack(fill=tk.X)

        tk.Label(ctrl, text="Split:").pack(side=tk.LEFT)
        self._split_var = tk.StringVar()
        split_cb = ttk.Combobox(ctrl, textvariable=self._split_var,
                                values=SPLITS, width=6, state="readonly")
        split_cb.pack(side=tk.LEFT, padx=(2, 12))
        split_cb.bind("<<ComboboxSelected>>",
                      lambda _: self._load_split(self._split_var.get()))

        tk.Button(ctrl, text="◄", width=3,
                  command=lambda: self._navigate(-1)).pack(side=tk.LEFT)
        tk.Button(ctrl, text="►", width=3,
                  command=lambda: self._navigate(+1)).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Rand",
                  command=self._random).pack(side=tk.LEFT, padx=(4, 12))

        self._idx_var = tk.StringVar()
        idx_entry = tk.Entry(ctrl, textvariable=self._idx_var, width=8)
        idx_entry.pack(side=tk.LEFT)
        idx_entry.bind("<Return>", lambda _: self._jump())

        self._total_lbl = tk.Label(ctrl, text="/ —")
        self._total_lbl.pack(side=tk.LEFT, padx=(2, 0))

        tk.Label(ctrl, text="  ← → : navigate   R : random",
                 fg="gray").pack(side=tk.RIGHT)

        # ── Image panels ─────────────────────────────────────────────────
        panels = tk.Frame(self)
        panels.pack(padx=8, pady=4)

        lf_orig = tk.LabelFrame(panels, text="Original  +  Annotations",
                                padx=4, pady=4)
        lf_orig.grid(row=0, column=0, padx=4)
        self._lbl_orig = tk.Label(lf_orig, width=PANEL_W, height=PANEL_H,
                                  bg="#1a1a1a")
        self._lbl_orig.pack()

        lf_heat = tk.LabelFrame(panels, text="Distance Heatmap  (hot colormap)",
                                padx=4, pady=4)
        lf_heat.grid(row=0, column=1, padx=4)
        self._lbl_heat = tk.Label(lf_heat, width=PANEL_W, height=PANEL_H,
                                  bg="#1a1a1a")
        self._lbl_heat.pack()

        # ── Info bar ─────────────────────────────────────────────────────
        self._info_var = tk.StringVar()
        tk.Label(self, textvariable=self._info_var, anchor="w",
                 justify=tk.LEFT, font=("Monospace", 9),
                 padx=10, pady=4).pack(fill=tk.X)

        # ── Seek bar ──────────────────────────────────────────────────────
        seek_frame = tk.Frame(self, padx=8, pady=4)
        seek_frame.pack(fill=tk.X)

        self._seek_start_lbl = tk.Label(seek_frame, text="0", width=6, anchor="e")
        self._seek_start_lbl.pack(side=tk.LEFT)

        self._seek_var = tk.DoubleVar(value=0)
        self._seekbar = ttk.Scale(
            seek_frame,
            from_=0, to=1,
            orient=tk.HORIZONTAL,
            variable=self._seek_var,
            command=self._on_seek,
        )
        self._seekbar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        self._seek_end_lbl = tk.Label(seek_frame, text="—", width=6, anchor="w")
        self._seek_end_lbl.pack(side=tk.LEFT)

        self._seek_after_id: str | None = None

    # ── Dataset operations ───────────────────────────────────────────────

    def _load_split(self, split: str):
        self._split_var.set(split)
        self._ds = SurgicalToolDataset(self._data_root, split)
        n = len(self._ds)
        self._total_lbl.config(text=f"/ {n}")
        self._seekbar.config(to=max(1, n - 1))
        self._seek_end_lbl.config(text=str(n - 1))
        self._show(0)

    def _navigate(self, delta: int):
        if self._ds:
            self._show((self._idx + delta) % len(self._ds))

    def _random(self):
        if self._ds:
            self._show(int(np.random.randint(0, len(self._ds))))

    def _jump(self):
        if not self._ds:
            return
        try:
            idx = int(self._idx_var.get())
            self._show(max(0, min(idx, len(self._ds) - 1)))
        except ValueError:
            pass

    def _on_seek(self, _value=None):
        """Debounced seek-bar handler — defers _show by 150 ms."""
        if self._seek_after_id is not None:
            self.after_cancel(self._seek_after_id)
        self._seek_after_id = self.after(150, self._apply_seek)

    def _apply_seek(self):
        self._seek_after_id = None
        if not self._ds:
            return
        idx = int(round(self._seek_var.get()))
        idx = max(0, min(idx, len(self._ds) - 1))
        if idx != self._idx:
            self._show(idx)

    # ── Render ───────────────────────────────────────────────────────────

    def _show(self, idx: int):
        if self._ds is None:
            return
        self._idx = idx
        self._idx_var.set(str(idx))
        self._seek_var.set(idx)

        image, target = self._ds[idx]       # ndarray uint8 (H,W,3), float32 (H,W)
        ann_path = self._ds.samples[idx]

        with open(ann_path) as f:
            ann_data = json.load(f)
        annotations = ann_data["annotations"]

        # Left: original with bbox + tip overlays
        left_img = _draw_annotations(image, annotations)

        # Right: heatmap blended onto image
        right_img = _blend_heatmap(image, target)

        # Keep references alive (required for tkinter PhotoImage)
        self._photos = [_to_photo(left_img), _to_photo(right_img)]
        self._lbl_orig.config(image=self._photos[0])
        self._lbl_heat.config(image=self._photos[1])

        # Info bar
        stem = os.path.splitext(os.path.basename(ann_path))[0]
        n = len(annotations)
        tips = "  ".join(
            f"tip[{i}]=({a['tip']['x']}, {a['tip']['y']})"
            for i, a in enumerate(annotations)
        )
        heat_range = (f"heatmap: [{target.min():.3f}, {target.max():.3f}]"
                      if target.max() > 0 else "heatmap: all zero (no tool)")
        self._info_var.set(
            f"File: {stem}   Tools: {n}   {heat_range}\n"
            + (tips if tips else "(no tools in this frame)")
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Browse SurgicalToolDataset with distance-based heatmap overlay"
    )
    parser.add_argument("--data-root", default="data/dataset",
                        help="Path to data/dataset/ directory")
    parser.add_argument("--split", default="train", choices=SPLITS,
                        help="Dataset split to open initially")
    args = parser.parse_args()

    app = DatasetBrowser(data_root=args.data_root, split=args.split)
    app.mainloop()


if __name__ == "__main__":
    main()
