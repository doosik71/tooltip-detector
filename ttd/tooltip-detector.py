#!/usr/bin/env python3
"""Interactive GUI for surgical tool tip detection.

Loads a trained TooltipDetector model and runs per-frame inference on dataset
splits or arbitrary image files.  Displays the original image with GT and
predicted tip overlays, the sigmoid heatmap, and accumulated evaluation metrics.

Panels
------
  Left  : original image + GT (colored circle/crosshair) + predicted tip (red X)
           yellow line = GT → nearest prediction
  Right : heatmap (hot colormap) + predicted peak markers

Controls
--------
  Threshold slider  : minimum heatmap value to count as a peak
  NMS radius slider : minimum pixel distance between kept peaks
  ← → arrow keys   : frame navigation
  R key             : random frame

Usage
-----
    uv run python ttd/tooltip-detector.py
    uv run python ttd/tooltip-detector.py --model-type monai_mini
    uv run python ttd/tooltip-detector.py --model data/models/monai/best.pt --data-root data/dataset

The model architecture can also be switched at runtime via the GUI's "Model"
dropdown, which loads data/models/<model-type>/best.pt for the selected type.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ttd.model import REGISTRY as MODEL_REGISTRY
from ttd.model import build as build_model
from ttd.train import _eval_transform
from ttd.eval import find_peaks
from ttd.dataset import SurgicalToolDataset


SPLITS = ["train", "val", "test"]
PANEL_W = 552   # 736 × 0.75
PANEL_H = 360   # 480 × 0.75

# GT annotations: one color per tool index
_TOOL_COLORS = [
    "#00FF00", "#FF8800", "#00AAFF", "#FF00FF",
    "#FFFF00", "#FF4444", "#44FFFF", "#FF44FF",
]
_PRED_COLOR  = "#FF3333"  # predicted tip X marker
_LINE_COLOR  = "#FFFF00"  # GT → prediction error line

_HIT_THRESHOLDS = (10, 20, 50)
_SEG_OVERLAY_RGB = np.array([0, 220, 120], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _default_model_path(model_type: str) -> str:
    return os.path.join("data", "models", model_type, "best.pt")


def _infer_model_type_from_path(model_path: str) -> str | None:
    norm = os.path.normpath(model_path)
    parts = norm.split(os.sep)
    for model_type in MODEL_REGISTRY:
        if model_type in parts:
            return model_type
    return None


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _colorize(heatmap: np.ndarray) -> np.ndarray:
    """Float32 [0,1] → RGB uint8 using a hot colormap (black→red→yellow→white)."""
    r = np.clip(heatmap * 3.0,       0.0, 1.0)
    g = np.clip(heatmap * 3.0 - 1.0, 0.0, 1.0)
    b = np.clip(heatmap * 3.0 - 2.0, 0.0, 1.0)
    return (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)


def _blend_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    colored = _colorize(heatmap)
    mask = (heatmap > 0)[..., np.newaxis]
    return np.where(mask,
                    (image * (1.0 - alpha) + colored * alpha).astype(np.uint8),
                    image)


def _blend_segmentation(
    image: np.ndarray,
    segmentation: np.ndarray | None,
    alpha: float = 0.35,
) -> np.ndarray:
    """Overlay a binary segmentation mask onto an RGB image."""
    if segmentation is None:
        return image

    mask = segmentation > 0
    if not mask.any():
        return image

    blended = image.copy()
    blended[mask] = (
        blended[mask] * (1.0 - alpha) + _SEG_OVERLAY_RGB * alpha
    ).astype(np.uint8)
    return blended


def _draw_gt(image: np.ndarray, annotations: list) -> np.ndarray:
    """Draw GT bounding boxes and tip markers (colored per tool)."""
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
        r = 8
        draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=color, outline="white")
        draw.line([tx - 12, ty, tx + 12, ty], fill="white", width=1)
        draw.line([tx, ty - 12, tx, ty + 12], fill="white", width=1)
    return np.array(pil)


def _draw_predictions(image: np.ndarray,
                      peaks: list,
                      gt_tips: list | None = None,
                      draw_X_markers: bool = True) -> np.ndarray:
    """Draw predicted tip X markers and yellow lines to GT tips."""
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)

    # Yellow lines: GT → nearest prediction
    if gt_tips and peaks:
        for gx, gy in gt_tips:
            best = min(peaks, key=lambda p: np.hypot(gx - p[0], gy - p[1]))
            cx, cy = int(best[0]), int(best[1])
            draw.line([gx, gy, cx, cy], fill=_LINE_COLOR, width=1)

    # Red X markers for predictions
    for px, py, _ in peaks:
        px, py = int(px), int(py)

        if draw_X_markers:
            r = 10
            draw.line([px - r, py - r, px + r, py + r], fill=_PRED_COLOR, width=6)
            draw.line([px + r, py - r, px - r, py + r], fill=_PRED_COLOR, width=6)

        # Small white dot at center
        s = 4
        draw.ellipse([px - s, py - s, px + s, py + s], fill="white")

    return np.array(pil)


def _to_photo(arr: np.ndarray) -> ImageTk.PhotoImage:
    pil = Image.fromarray(arr).resize((PANEL_W, PANEL_H), Image.BILINEAR)
    return ImageTk.PhotoImage(pil)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class TooltipDetectorApp(tk.Tk):
    def __init__(self, model_type: str, model_path: str, data_root: str):
        super().__init__()
        self.title("Tooltip Detector")
        self.resizable(True, True)

        self._data_root  = data_root
        self._model_type = model_type
        self._model_path = model_path
        self._device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform  = _eval_transform()

        self._ds: SurgicalToolDataset | None = None
        self._idx = 0
        self._photos: list[ImageTk.PhotoImage] = []
        self._current_file_image: np.ndarray | None = None  # file mode: 480×736 RGB

        self._playing = False
        self._play_after_id: str | None = None

        self._stats_reset()
        self._model = self._load_model(model_type, model_path)

        self._build_ui()
        self._load_split("test")

        # Lock in the natural layout size as the minimum so per-frame result
        # updates (info/stats text) never grow or shrink the window; the user
        # can still enlarge it freely via the normal resize handles.
        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())

        self.bind("<Left>",  lambda _: self._navigate(-1))
        self.bind("<Right>", lambda _: self._navigate(+1))
        self.bind("<r>",     lambda _: self._random())

    # ── Model ────────────────────────────────────────────────────────────

    def _load_model(self, model_type: str, path: str) -> torch.nn.Module | None:
        try:
            model = build_model(model_type, num_classes=2).to(self._device)
            state = torch.load(path, map_location=self._device, weights_only=False)
            model.load_state_dict(state)
            model.eval()
            print(f"Model loaded: {model_type}  {path}  [{self._device}]")
            return model
        except Exception as exc:
            print(f"WARNING: could not load model '{model_type}' from {path} — {exc}")
            return None

    def _apply_model(self, model_type: str, model_path: str, *, show_error: bool):
        self._model_type = model_type
        self._model_path = model_path
        self._model = self._load_model(model_type, model_path)
        self._refresh_model_status()

        if show_error and self._model is None:
            messagebox.showwarning(
                "Model Load Warning",
                f"Could not load model '{model_type}' from:\n{model_path}",
            )

    def _refresh_model_status(self):
        if not hasattr(self, "_model_status_var"):
            return

        model_name = os.path.basename(self._model_path)
        if self._model:
            text = f"Model: {self._model_type} ({model_name})  [{self._device}]"
            color = "#005500"
        else:
            text = f"Model NOT loaded: {self._model_type} ({model_name})"
            color = "#aa0000"

        self._model_status_var.set(text)
        self._model_status_lbl.config(fg=color)

    def _rerender_current_view(self):
        if self._mode_var.get() == "dataset" and self._ds:
            self._render_dataset_frame(self._idx, update_stats=False)
        elif self._current_file_image is not None:
            self._render_file_frame(self._current_file_image)

    def _load_segmentation_mask(self, ann_path: str) -> np.ndarray | None:
        if self._ds is None:
            return None

        stem = os.path.splitext(os.path.basename(ann_path))[0]
        seg_path = os.path.join(self._ds.seg_dir, stem + ".png")
        if not os.path.exists(seg_path):
            return None

        try:
            return np.array(Image.open(seg_path))
        except Exception as exc:
            print(f"WARNING: could not load segmentation mask {seg_path} — {exc}")
            return None

    def _infer(self, image: np.ndarray) -> tuple[np.ndarray, list, float]:
        """Run the model on a raw HWC uint8 RGB image.

        Returns (heatmap, peaks, infer_ms) where heatmap is float32 (480, 736),
        peaks is a list of (x, y, confidence) tuples, and infer_ms is the model
        forward time in milliseconds.
        """
        if self._model is None:
            return np.zeros((480, 736), dtype=np.float32), [], 0.0

        tensor = self._transform(image=image)["image"]
        tensor = tensor.unsqueeze(0).to(self._device, dtype=torch.float32)

        if self._device.type == "cuda":
            torch.cuda.synchronize()
        infer_t0 = time.perf_counter()
        with torch.no_grad():
            pred = self._model(tensor)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - infer_t0) * 1000.0

        heatmap = torch.sigmoid(pred[0, 1]).cpu().numpy()  # (H, W)

        threshold  = self._threshold_var.get()
        nms_radius = int(self._nms_var.get())
        peaks = find_peaks(heatmap, threshold, nms_radius)

        return heatmap, peaks, infer_ms

    # ── Accumulated stats ────────────────────────────────────────────────

    def _stats_reset(self):
        self._n_frames  = 0
        self._n_gt_tips = 0
        self._n_missed  = 0
        self._all_dists: list[float] = []
        self._hits = {t: 0 for t in _HIT_THRESHOLDS}

    def _stats_update(self, gt_tips: list, peaks: list):
        if not gt_tips:
            return
        self._n_frames += 1
        for gx, gy in gt_tips:
            self._n_gt_tips += 1
            if not peaks:
                self._n_missed += 1
                continue
            best = min(peaks, key=lambda p: np.hypot(gx - p[0], gy - p[1]))
            dist = float(np.hypot(gx - best[0], gy - best[1]))
            self._all_dists.append(dist)
            for t in _HIT_THRESHOLDS:
                if dist <= t:
                    self._hits[t] += 1

    def _stats_str(self) -> str:
        n     = self._n_gt_tips
        safe  = max(1, n)
        arr   = np.array(self._all_dists) if self._all_dists else np.zeros(0)

        mean_d = f"{arr.mean():.2f} px"          if arr.size else "—"
        med_d  = f"{np.median(arr):.2f} px"      if arr.size else "—"
        p90_d  = f"{np.percentile(arr, 90):.2f} px" if arr.size else "—"
        miss_r = f"{self._n_missed / safe * 100:.1f}%" if n else "—"
        hits   = {t: (f"{self._hits[t] / safe * 100:.1f}%" if n else "—")
                  for t in _HIT_THRESHOLDS}

        return (
            f"Frames: {self._n_frames:>5}   GT tips: {n:>6}   "
            f"Missed: {self._n_missed:>5} ({miss_r})\n"
            f"Mean dist: {mean_d:<12}  Median: {med_d:<12}  P90: {p90_d}\n"
            f"Hit@10px: {hits[10]:<9}  Hit@20px: {hits[20]:<9}  Hit@50px: {hits[50]}"
        )

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        # ── Row 1: mode / split / file / navigation ───────────────────────
        ctrl = tk.Frame(self, pady=6, padx=8)
        ctrl.pack(fill=tk.X)

        tk.Label(ctrl, text="Mode:").pack(side=tk.LEFT)
        self._mode_var = tk.StringVar(value="dataset")
        mode_cb = ttk.Combobox(ctrl, textvariable=self._mode_var,
                               values=["dataset", "file"], width=8, state="readonly")
        mode_cb.pack(side=tk.LEFT, padx=(2, 8))
        mode_cb.bind("<<ComboboxSelected>>", self._on_mode_change)

        tk.Label(ctrl, text="Split:").pack(side=tk.LEFT)
        self._split_var = tk.StringVar()
        self._split_cb = ttk.Combobox(ctrl, textvariable=self._split_var,
                                      values=SPLITS, width=6, state="readonly")
        self._split_cb.pack(side=tk.LEFT, padx=(2, 8))
        self._split_cb.bind("<<ComboboxSelected>>",
                            lambda _: self._load_split(self._split_var.get()))

        tk.Label(ctrl, text="Model:").pack(side=tk.LEFT)
        self._model_var = tk.StringVar(value=self._model_type)
        self._model_cb = ttk.Combobox(
            ctrl,
            textvariable=self._model_var,
            values=list(MODEL_REGISTRY),
            width=12,
            state="readonly",
        )
        self._model_cb.pack(side=tk.LEFT, padx=(2, 8))
        self._model_cb.bind("<<ComboboxSelected>>", self._on_model_change)

        self._file_btn = tk.Button(ctrl, text="Open Image...",
                                   command=self._open_file, state=tk.DISABLED)
        self._file_btn.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(ctrl, text="<-", width=3,
                  command=lambda: self._navigate(-1)).pack(side=tk.LEFT)
        tk.Button(ctrl, text="->", width=3,
                  command=lambda: self._navigate(+1)).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Rand",
                  command=self._random).pack(side=tk.LEFT, padx=(4, 8))

        self._play_btn = tk.Button(ctrl, text="Play", width=6, command=self._play)
        self._play_btn.pack(side=tk.LEFT, padx=(0, 2))
        self._pause_btn = tk.Button(ctrl, text="Pause", width=6,
                                    command=self._pause, state=tk.DISABLED)
        self._pause_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._idx_var = tk.StringVar()
        idx_entry = tk.Entry(ctrl, textvariable=self._idx_var, width=8)
        idx_entry.pack(side=tk.LEFT)
        idx_entry.bind("<Return>", lambda _: self._jump())

        self._total_lbl = tk.Label(ctrl, text="/ —", width=10, anchor="w")
        self._total_lbl.pack(side=tk.LEFT, padx=(2, 0))

        tk.Label(ctrl, text="  <- -> : navigate   R : random",
                 fg="gray").pack(side=tk.RIGHT)

        # ── Row 2: threshold / NMS sliders + model info ───────────────────
        param = tk.Frame(self, pady=4, padx=8)
        param.pack(fill=tk.X)

        tk.Label(param, text="Threshold:").pack(side=tk.LEFT)
        self._threshold_var = tk.DoubleVar(value=0.5)
        ttk.Scale(param, from_=0.05, to=0.95, orient=tk.HORIZONTAL,
                  variable=self._threshold_var, length=150,
                  command=lambda _: self._on_param_change()
                  ).pack(side=tk.LEFT, padx=(4, 2))
        self._thr_lbl = tk.Label(param, text="0.50", width=5)
        self._thr_lbl.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(param, text="NMS radius:").pack(side=tk.LEFT)
        self._nms_var = tk.DoubleVar(value=20)
        ttk.Scale(param, from_=5, to=50, orient=tk.HORIZONTAL,
                  variable=self._nms_var, length=150,
                  command=lambda _: self._on_param_change()
                  ).pack(side=tk.LEFT, padx=(4, 2))
        self._nms_lbl = tk.Label(param, text=" 20px", width=5)
        self._nms_lbl.pack(side=tk.LEFT, padx=(0, 16))

        self._model_status_var = tk.StringVar()
        self._model_status_lbl = tk.Label(
            param,
            textvariable=self._model_status_var,
            font=("Monospace", 9),
            width=48, anchor="w",
        )
        self._model_status_lbl.pack(side=tk.LEFT)
        self._refresh_model_status()

        # ── Row 3: two image panels ───────────────────────────────────────
        panels = tk.Frame(self)
        panels.pack(padx=8, pady=4)

        lf_orig = tk.LabelFrame(
            panels,
            text="Original  [ GT: colored o  Pred: red x  Error: yellow line ]",
            padx=4, pady=4,
        )
        lf_orig.grid(row=0, column=0, padx=4)
        self._lbl_orig = tk.Label(lf_orig, width=PANEL_W, height=PANEL_H, bg="#1a1a1a")
        self._lbl_orig.pack()

        lf_heat = tk.LabelFrame(
            panels,
            text="Heatmap (hot colormap)  +  Predicted peaks",
            padx=4, pady=4,
        )
        lf_heat.grid(row=0, column=1, padx=4)
        self._lbl_heat = tk.Label(lf_heat, width=PANEL_W, height=PANEL_H, bg="#1a1a1a")
        self._lbl_heat.pack()

        # ── Row 4: per-frame info (fixed height; scrolls instead of resizing
        #           the window when a frame has many GT tips / predictions) ──
        info_frame = tk.Frame(self, padx=8, pady=2)
        info_frame.pack(fill=tk.BOTH, expand=True)

        info_scroll = tk.Scrollbar(info_frame, orient=tk.VERTICAL)
        self._info_text = tk.Text(
            info_frame, height=5, wrap=tk.WORD, font=("Monospace", 9),
            relief=tk.FLAT, bg=self.cget("bg"), state=tk.DISABLED,
            yscrollcommand=info_scroll.set,
        )
        info_scroll.config(command=self._info_text.yview)
        self._info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Row 5: seek bar ───────────────────────────────────────────────
        seek_frame = tk.Frame(self, padx=8, pady=2)
        seek_frame.pack(fill=tk.X)

        tk.Label(seek_frame, text="0", width=6, anchor="e").pack(side=tk.LEFT)
        self._seek_var = tk.DoubleVar(value=0)
        self._seekbar  = ttk.Scale(seek_frame, from_=0, to=1,
                                   orient=tk.HORIZONTAL, variable=self._seek_var,
                                   command=self._on_seek)
        self._seekbar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self._seek_end_lbl = tk.Label(seek_frame, text="—", width=6, anchor="w")
        self._seek_end_lbl.pack(side=tk.LEFT)
        self._seek_after_id: str | None = None

        # ── Row 6: accumulated stats panel ───────────────────────────────
        stats_outer = tk.Frame(self, pady=4, padx=8)
        stats_outer.pack(fill=tk.X)

        lf_stats = tk.LabelFrame(stats_outer,
                                 text="Accumulated Evaluation  (dataset mode, frames with GT tips)",
                                 padx=8, pady=6)
        lf_stats.pack(fill=tk.X)

        tk.Button(lf_stats, text="Reset Stats",
                  command=self._reset_stats_ui).pack(side=tk.RIGHT, padx=4)

        self._stats_var = tk.StringVar()
        tk.Label(lf_stats, textvariable=self._stats_var, anchor="w",
                 justify=tk.LEFT, font=("Monospace", 9)).pack(side=tk.LEFT)

        self._refresh_stats_display()

    def _set_info(self, text: str):
        """Update the per-frame info panel without resizing the window."""
        self._info_text.config(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)
        self._info_text.insert(tk.END, text)
        self._info_text.config(state=tk.DISABLED)

    # ── Mode / parameter handlers ────────────────────────────────────────

    def _on_mode_change(self, _=None):
        self._pause()
        mode = self._mode_var.get()
        if mode == "dataset":
            self._split_cb.config(state="readonly")
            self._file_btn.config(state=tk.DISABLED)
            self._play_btn.config(state=tk.NORMAL)
            self._load_split(self._split_var.get() or "test")
        else:
            self._split_cb.config(state=tk.DISABLED)
            self._file_btn.config(state=tk.NORMAL)
            self._play_btn.config(state=tk.DISABLED)
            # Clear panels
            self._lbl_orig.config(image="")
            self._lbl_heat.config(image="")
            self._set_info("[File mode]  Click 'Open Image…' to load an image.")

    # ── Playback ─────────────────────────────────────────────────────────

    def _play(self):
        if self._mode_var.get() != "dataset" or not self._ds or self._playing:
            return
        self._playing = True
        self._play_btn.config(state=tk.DISABLED)
        self._pause_btn.config(state=tk.NORMAL)
        self._schedule_play_tick()

    def _pause(self):
        self._playing = False
        if self._play_after_id is not None:
            self.after_cancel(self._play_after_id)
            self._play_after_id = None
        self._play_btn.config(state=tk.NORMAL)
        self._pause_btn.config(state=tk.DISABLED)

    def _schedule_play_tick(self):
        self._play_after_id = self.after(500, self._play_tick)

    def _play_tick(self):
        if not self._playing:
            return
        # Schedule the next tick before rendering so the 0.5s cadence isn't
        # inflated by this frame's inference/render time.
        self._schedule_play_tick()
        self._navigate(+1)

    def _on_model_change(self, _=None):
        model_type = self._model_var.get()
        model_path = _default_model_path(model_type)
        self._apply_model(model_type, model_path, show_error=True)
        self._stats_reset()
        self._refresh_stats_display()
        self._rerender_current_view()

    def _on_param_change(self):
        """Slider moved — update labels and re-render current frame (no stat update)."""
        thr = self._threshold_var.get()
        nms = int(self._nms_var.get())
        self._thr_lbl.config(text=f"{thr:.2f}")
        self._nms_lbl.config(text=f"{nms:>3}px")

        self._rerender_current_view()

    # ── Dataset operations ───────────────────────────────────────────────

    def _load_split(self, split: str):
        self._split_var.set(split)
        self._ds = SurgicalToolDataset(self._data_root, split)
        n = len(self._ds)
        self._total_lbl.config(text=f"/ {n}")
        self._seekbar.config(to=max(1, n - 1))
        self._seek_end_lbl.config(text=str(n - 1))
        self._stats_reset()
        self._refresh_stats_display()
        self._show(0)

    def _show(self, idx: int):
        """Navigate to dataset frame idx and update stats."""
        if self._ds is None:
            return
        self._idx = idx
        self._idx_var.set(str(idx))
        self._seek_var.set(idx)
        self._render_dataset_frame(idx, update_stats=True)

    def _navigate(self, delta: int):
        if self._mode_var.get() == "dataset" and self._ds:
            self._show((self._idx + delta) % len(self._ds))

    def _random(self):
        if self._mode_var.get() == "dataset" and self._ds:
            self._show(int(np.random.randint(0, len(self._ds))))

    def _jump(self):
        if not self._ds:
            return
        try:
            idx = int(self._idx_var.get())
            self._show(max(0, min(idx, len(self._ds) - 1)))
        except ValueError:
            pass

    def _on_seek(self, _=None):
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

    # ── File mode ────────────────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        try:
            raw = np.array(Image.open(path).convert("RGB"))
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            return

        # Resize to model input size so prediction coords align with display
        image = np.array(Image.fromarray(raw).resize((736, 480), Image.BILINEAR))
        self._current_file_image = image
        self._idx_var.set(os.path.basename(path))
        self._total_lbl.config(text="")
        self._render_file_frame(image)

    # ── Rendering ────────────────────────────────────────────────────────

    def _render_dataset_frame(self, idx: int, *, update_stats: bool):
        if self._ds is None:
            return

        image, _ = self._ds[idx]          # HWC uint8 numpy (no transform applied)
        ann_path  = self._ds.samples[idx]

        with open(ann_path) as f:
            ann_data = json.load(f)
        annotations = ann_data["annotations"]
        gt_tips = [(a["tip"]["x"], a["tip"]["y"]) for a in annotations]
        segmentation = self._load_segmentation_mask(ann_path)

        heatmap, peaks, infer_ms = self._infer(image)

        if update_stats:
            self._stats_update(gt_tips, peaks)
            self._refresh_stats_display()

        # Left panel: segmentation mask + GT + predictions + error lines
        left = _blend_segmentation(image.copy(), segmentation)
        left = _draw_gt(left, annotations)
        left = _draw_predictions(left, peaks, gt_tips, False)

        # Right panel: heatmap + predicted peaks
        right = _blend_heatmap(image, heatmap)
        right = _draw_predictions(right, peaks)

        self._set_panels(left, right)
        self._set_info_dataset(ann_path, gt_tips, peaks, infer_ms)

    def _render_file_frame(self, image: np.ndarray):
        heatmap, peaks, infer_ms = self._infer(image)

        left  = _draw_predictions(image.copy(), peaks)
        right = _blend_heatmap(image, heatmap)
        right = _draw_predictions(right, peaks)

        self._set_panels(left, right)

        peaks_info = "  ".join(
            f"({int(p[0])},{int(p[1])}) conf={p[2]:.3f}" for p in peaks
        )
        self._set_info(
            f"[File mode]  Detected tips: {len(peaks)}\n"
            f"Inference time: {infer_ms:.2f} ms\n"
            + (peaks_info if peaks_info else "(none — try lowering threshold)")
        )

    def _set_panels(self, left: np.ndarray, right: np.ndarray):
        self._photos = [_to_photo(left), _to_photo(right)]
        self._lbl_orig.config(image=self._photos[0])
        self._lbl_heat.config(image=self._photos[1])

    def _set_info_dataset(
        self,
        ann_path: str,
        gt_tips: list,
        peaks: list,
        infer_ms: float,
    ):
        stem  = os.path.splitext(os.path.basename(ann_path))[0]
        lines = [
            f"File: {stem}   GT tips: {len(gt_tips)}   Predicted: {len(peaks)}",
            f"Inference time: {infer_ms:.2f} ms",
        ]

        if gt_tips and peaks:
            for i, (gx, gy) in enumerate(gt_tips):
                best = min(peaks, key=lambda p: np.hypot(gx - p[0], gy - p[1]))
                dist = float(np.hypot(gx - best[0], gy - best[1]))
                hit_str = ""
                for t in _HIT_THRESHOLDS:
                    if dist <= t:
                        hit_str = f"  v hit@{t}px"
                        break
                else:
                    hit_str = "  x miss"
                lines.append(
                    f"  tip[{i}] GT=({gx},{gy})  pred=({int(best[0])},{int(best[1])})  "
                    f"dist={dist:.1f}px  conf={best[2]:.3f}{hit_str}"
                )
        elif gt_tips and not peaks:
            lines.append("  (no predictions — try lowering threshold or NMS radius)")

        self._set_info("\n".join(lines))

    # ── Stats display ─────────────────────────────────────────────────────

    def _refresh_stats_display(self):
        self._stats_var.set(self._stats_str())

    def _reset_stats_ui(self):
        self._stats_reset()
        self._refresh_stats_display()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive GUI for surgical tool tip detection"
    )
    parser.add_argument("--model-type", default="monai",
                        choices=list(MODEL_REGISTRY),
                        help="Model architecture to load (default: monai)")
    parser.add_argument("--model",     default=None,
                        help="Path to trained model weights "
                             "(default: data/models/<model-type>/best.pt)")
    parser.add_argument("--data-root", default="data/dataset",
                        help="Path to data/dataset/ directory (default: data/dataset)")
    args = parser.parse_args()

    model_path = args.model or _default_model_path(args.model_type)
    model_type = args.model_type
    inferred_model_type = _infer_model_type_from_path(model_path)
    if args.model is not None and inferred_model_type is not None:
        model_type = inferred_model_type

    app = TooltipDetectorApp(
        model_type=model_type,
        model_path=model_path,
        data_root=args.data_root,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
