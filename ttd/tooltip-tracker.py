#!/usr/bin/env python3
"""Interactive GUI for real-time surgical tool tip tracking in video.

Loads a trained TooltipDetector model (heatmap-based, same architecture as
tooltip-detector.py) and runs per-frame inference on a user-selected video
file.  On each frame, an arrow is drawn from the frame center toward the
detected tool tip.  The arrow represents the direction the endoscope camera
should move, so its direction and length are only ever allowed to change
gradually -- a sudden jump would translate into a sudden, unsafe robot
motion.

Arrow smoothing
----------------
The arrow's length/direction is owned by ArrowState, which turns each
frame's raw peaks into a single measurement (resolve_measurement, below)
and feeds it to one of three interchangeable ttd.camera_motion_vector
implementations, selectable at runtime via the "Method" dropdown:
CameraMotionVectorMagnitudeBlend (default -- smooths length only,
direction snaps to the measurement), CameraMotionVectorBlend (smooths
the full vector, length and direction together), and
CameraMotionVectorKalman (a Kalman filter, kept for comparison). See
that module for the full rationale of each: why the tip itself isn't
tracked directly, and why the blend/noise rate differs for "keep going"
vs "stop".

False-positive guards
----------------------
  - 3 or more tips detected in a frame -> treated as "not detected".
  - Exactly 2 tips detected -> if the angle between the center->tip vectors
    is >= 90 degrees, treated as "not detected" (ambiguous / spurious);
    otherwise the two vectors are averaged into a single measurement.

Arrow color
-----------
  Pink  (분홍색) : zero tips detected this frame.
  Green (연두색) : one or more tips detected this frame -- including counts
                   that get filtered out below as likely false positives.
                   The tip-count polygon and the fast "stop" blend rate
                   already flag those cases; the arrow color only
                   distinguishes "nothing found" from "something found".

Controls
--------
  Open Video...     : choose a video file
  Method            : choose the arrow-smoothing implementation (see
                       "Arrow smoothing" above); switching resets the
                       arrow to zero since internal state isn't portable
                       between implementations
  Play / Pause       : continuous, sequential playback with best-effort
                        real-time pacing (frames are dropped, never
                        reordered or skipped-then-reprocessed, to keep up)
  <- ->              : single-frame step (always sequential)
  Seek bar            : jump to an arbitrary frame (manual scrub)
  Threshold / NMS     : same peak-extraction parameters as tooltip-detector

Each method's own internal blend/noise rates (W1/W2, or Kalman's process
noise) are fixed in ttd.camera_motion_vector -- only which method runs
is a GUI control, not its internal tuning.

Usage
-----
    uv run python ttd/tooltip-tracker.py
    uv run python ttd/tooltip-tracker.py --model-type monai_mini
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk


# NOTE: deliberately not top-level imports. `ttd.*` requires the project
# root on sys.path, which only exists after the insert() below runs.
# Editor/linter "organize imports" actions only reorder top-level import
# statements, so nesting these under `if True:` keeps them pinned after the
# sys.path fix-up instead of being hoisted back above it on every save.
if True:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from ttd.camera_motion_vector import (
        CameraMotionVectorMagnitudeBlend,
        CameraMotionVectorBlend,
        CameraMotionVectorKalman,
        DEFAULT_PROCESS_VAR_GROW,
        DEFAULT_MEASUREMENT_VAR,
    )
    from ttd.eval import find_peaks
    from ttd.train import _eval_transform
    from ttd.model import build as build_model
    from ttd.model import REGISTRY as MODEL_REGISTRY


FRAME_W, FRAME_H = 736, 480          # model input size, and the coordinate
# space peaks/arrows are computed in
CENTER = (FRAME_W / 2.0, FRAME_H / 2.0)

# on-screen display size (736 x 480 * 0.75,
PANEL_W, PANEL_H = 552, 360
# same scale-down as tooltip-detector.py so
# the window fits on smaller screens)

_MAX_TIPS_BEFORE_REJECT = 3           # >= this many peaks => treat as no detection
_TWO_TIP_MAX_ANGLE_DEG = 90.0         # >= this angle between two tips => reject
# px, visual clamp only (state is unclamped)
_MAX_ARROW_DRAW_LEN = 200.0
_MIN_ARROW_DRAW_LEN = 2.0             # px, below this nothing is drawn

# > this many peaks => draw the warning polygon
_TIP_OVERFLOW_THRESHOLD = 3
_MAX_POLYGON_SIDES = 5                # capped at a pentagon -- a hexagon reads too
# much like a circle to tell apart at a glance
_POLYGON_RADIUS = 12.0                # px

_MIN_VECTOR_LEN = 5.0

_COLOR_DETECTED = "#8CFF3C"         # 연두색 (yellow-green)
_COLOR_NOT_DETECTED = "#FF7FCF"     # 분홍색 (pink)
_COLOR_RAW_MARKER = "#33AAFF"
_COLOR_POLYGON_OUTLINE = "#EE1111"  # 붉은색 (red), outline only -- no fill

# Selectable camera-motion smoothing implementations (ttd.camera_motion_vector).
# Kalman needs constructor args the other two don't, so each entry is a
# no-arg factory rather than the class itself.
_METHOD_FACTORIES = {
    "CameraMotionVectorMagnitudeBlend": lambda: CameraMotionVectorMagnitudeBlend(),
    "CameraMotionVectorBlend": lambda: CameraMotionVectorBlend(),
    "CameraMotionVectorKalman": lambda: CameraMotionVectorKalman(
        DEFAULT_PROCESS_VAR_GROW, DEFAULT_MEASUREMENT_VAR),
}
_DEFAULT_METHOD = "CameraMotionVectorMagnitudeBlend"


def _build_motion_vector(method: str):
    return _METHOD_FACTORIES[method]()


# ---------------------------------------------------------------------------
# Detection -> single "camera direction" measurement
# ---------------------------------------------------------------------------

def resolve_measurement(
    peaks: list[tuple[float, float, float]],
    center: tuple[float, float] = CENTER,
) -> tuple[tuple[float, float], bool, str, bool]:
    """Turn raw peaks into a single (dx, dy) measurement + validity flag.

    Returns (measurement, valid, reason, two_tip_conflict). measurement is
    always a concrete (dx, dy) -- the zero vector when invalid, so the
    caller can feed it straight into the Kalman filter unconditionally.
    two_tip_conflict is True only for the "exactly 2 tips, too far apart in
    direction" case.
    """
    cx, cy = center

    if len(peaks) == 0:
        return (0.0, 0.0), False, "no tip detected", False

    if len(peaks) >= _MAX_TIPS_BEFORE_REJECT:
        return (0.0, 0.0), False, f"{len(peaks)} tips detected (>=3, likely false positive)", False

    if len(peaks) == 1:
        px, py, _ = peaks[0]
        return (px - cx, py - cy), True, "1 tip detected", False

    # Exactly two peaks: reject if their directions from center diverge too much.
    (p0x, p0y, _), (p1x, p1y, _) = peaks[0], peaks[1]
    v0 = np.array([p0x - cx, p0y - cy])
    v1 = np.array([p1x - cx, p1y - cy])
    n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
    if n0 < _MIN_VECTOR_LEN or n1 < _MIN_VECTOR_LEN:
        return (0.0, 0.0), False, "2 tips detected, one at center (degenerate)", False

    cos_angle = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))

    if angle_deg >= _TWO_TIP_MAX_ANGLE_DEG:
        return ((0.0, 0.0), False,
                f"2 tips detected, {angle_deg:.0f}° apart (>=90°, ambiguous)", True)

    avg = (v0 + v1) / 2.0
    return ((float(avg[0]), float(avg[1])), True,
            f"2 tips detected, {angle_deg:.0f}° apart (averaged)", False)


# ---------------------------------------------------------------------------
# Arrow state: owns the current length/direction and how it evolves per frame
# ---------------------------------------------------------------------------

class ArrowState:
    """Current camera-direction arrow (length + direction), and the logic
    that updates it from each frame's raw detected tips.

    This is the single entry point the GUI talks to: feed it a frame's
    peaks via update(), read back .vector / .length / .angle_deg for
    drawing, and .reason / .tip_count for the info panel. Internally it
    turns peaks into a single measurement (resolve_measurement) and feeds
    whichever ttd.camera_motion_vector implementation is selected
    (see _METHOD_FACTORIES) -- swap it at runtime with set_method().
    """

    def __init__(self, method: str = _DEFAULT_METHOD, center: tuple[float, float] = CENTER):
        self._center = center
        self.method = method
        self._kf = _build_motion_vector(method)
        self.reason = "no tip detected"
        self.tip_count = 0
        self.last_measurement = (0.0, 0.0)
        self.two_tip_conflict = False

    def _reset_tracking_state(self):
        self.reason = "no tip detected"
        self.tip_count = 0
        self.last_measurement = (0.0, 0.0)
        self.two_tip_conflict = False

    def reset(self):
        self._kf.reset()
        self._reset_tracking_state()

    def set_method(self, method: str):
        """Switch the smoothing implementation. The different
        implementations keep incompatible internal state (e.g. Kalman's
        covariance vs. a bare vector), so switching restarts the arrow
        from zero rather than trying to carry state across."""
        self.method = method
        self._kf = _build_motion_vector(method)
        self._reset_tracking_state()

    @property
    def vector(self) -> tuple[float, float]:
        return float(self._kf.x[0]), float(self._kf.x[1])

    @property
    def length(self) -> float:
        return float(np.hypot(*self.vector))

    @property
    def angle_deg(self) -> float:
        dx, dy = self.vector
        return float(np.degrees(np.arctan2(dy, dx)))

    def update(self, peaks: list[tuple[float, float, float]]) -> tuple[float, float]:
        """Advance the arrow state from this frame's detected peaks."""
        measurement, valid, reason, two_tip_conflict = resolve_measurement(
            peaks, self._center)
        self.reason = reason
        self.tip_count = len(peaks)
        self.last_measurement = measurement
        self.two_tip_conflict = two_tip_conflict
        return self._kf.step(measurement, valid)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _default_model_path(model_type: str) -> str:
    return os.path.join("data", "models", model_type, "best.pt")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_arrow(
    image: np.ndarray,
    vector: tuple[float, float],
    color: str,
    center: tuple[float, float] = CENTER,
) -> np.ndarray:
    """Draw an arrow from *center* toward *vector*, clamped for readability."""
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)

    cx, cy = center
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4],
                 fill="white", outline="black")

    length = float(np.hypot(*vector))
    if length >= _MIN_ARROW_DRAW_LEN:
        draw_len = min(length, _MAX_ARROW_DRAW_LEN)
        ux, uy = vector[0] / length, vector[1] / length
        tip_x, tip_y = cx + ux * draw_len, cy + uy * draw_len

        draw.line([cx, cy, tip_x, tip_y], fill=color, width=8)

        # Arrowhead: one filled triangle. Two separate thick strokes fanning
        # out from the tip read as two overlapping arrows at a glance; a
        # single solid shape is unambiguous.
        head_len, head_ang = 22.0, np.radians(22.0)
        base_ang = np.arctan2(uy, ux)
        wings = []
        for sign in (-1, 1):
            ang = base_ang + np.pi - sign * head_ang
            wings.append((tip_x + head_len * np.cos(ang),
                         tip_y + head_len * np.sin(ang)))
        draw.polygon([(tip_x, tip_y), wings[0], wings[1]], fill=color)

    return np.array(pil)


def _draw_raw_markers(image: np.ndarray, peaks: list[tuple[float, float, float]]) -> np.ndarray:
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    r = 10
    for px, py, _ in peaks:
        draw.ellipse([px - r, py - r, px + r, py + r],
                     outline=_COLOR_RAW_MARKER, width=4)
    return np.array(pil)


def _draw_tip_overflow_polygon(
    image: np.ndarray,
    num_tips: int,
    center: tuple[float, float] = CENTER,
) -> np.ndarray:
    """Warn about a likely tip-count false positive: a red n-gon at screen
    center, where n = num_tips capped at a pentagon (_MAX_POLYGON_SIDES)."""
    sides = min(num_tips, _MAX_POLYGON_SIDES)
    cx, cy = center
    points = []
    for i in range(sides):
        angle = -np.pi / 2 + i * (2 * np.pi / sides)
        points.append((cx + _POLYGON_RADIUS * np.cos(angle),
                       cy + _POLYGON_RADIUS * np.sin(angle)))

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.polygon(points, fill=None, outline=_COLOR_POLYGON_OUTLINE, width=6)
    return np.array(pil)


def _draw_two_tip_conflict_x(
    image: np.ndarray,
    center: tuple[float, float] = CENTER,
) -> np.ndarray:
    """Substitute for the tip-count polygon when exactly 2 tips are detected
    but too far apart in direction to average -- there's no such thing as a
    2-sided polygon, so a red X at screen center stands in for it."""
    cx, cy = center
    r = _POLYGON_RADIUS
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.line([cx - r, cy - r, cx + r, cy + r],
              fill=_COLOR_POLYGON_OUTLINE, width=6)
    draw.line([cx - r, cy + r, cx + r, cy - r],
              fill=_COLOR_POLYGON_OUTLINE, width=6)
    return np.array(pil)


def _to_photo(arr: np.ndarray) -> ImageTk.PhotoImage:
    pil = Image.fromarray(arr).resize((PANEL_W, PANEL_H), Image.BILINEAR)
    return ImageTk.PhotoImage(pil)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class TooltipTrackerApp(tk.Tk):
    def __init__(self, model_type: str, model_path: str):
        super().__init__()
        self.title("Tooltip Tracker")
        self.resizable(True, True)

        self._model_type = model_type
        self._model_path = model_path
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._transform = _eval_transform()

        self._cap: cv2.VideoCapture | None = None
        self._video_path: str | None = None
        self._fps = 30.0
        self._total_frames = 0
        self._photo: ImageTk.PhotoImage | None = None

        self._arrow = ArrowState()

        self._playing = False
        self._play_after_id: str | None = None
        self._play_start_wall: float = 0.0
        self._play_start_frame: int = 0

        self._model = self._load_model(model_type, model_path)

        self._build_ui()

        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())

        self.bind("<Left>", lambda _: self._step(-1))
        self.bind("<Right>", lambda _: self._step(+1))

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Model ────────────────────────────────────────────────────────────

    def _load_model(self, model_type: str, path: str) -> torch.nn.Module | None:
        try:
            model = build_model(model_type, num_classes=2).to(self._device)
            state = torch.load(
                path, map_location=self._device, weights_only=False)
            model.load_state_dict(state)
            model.eval()
            print(f"Model loaded: {model_type}  {path}  [{self._device}]")
            return model
        except Exception as exc:
            print(
                f"WARNING: could not load model '{model_type}' from {path} — {exc}")
            return None

    def _on_model_change(self, _=None):
        model_type = self._model_var.get()
        model_path = _default_model_path(model_type)
        self._model_type = model_type
        self._model_path = model_path
        self._model = self._load_model(model_type, model_path)
        self._refresh_model_status()
        if self._model is None:
            messagebox.showwarning(
                "Model Load Warning",
                f"Could not load model '{model_type}' from:\n{model_path}",
            )
        self._arrow.reset()
        self._render_current_frame(advance_kf=False)

    def _on_method_change(self, _=None):
        self._arrow.set_method(self._method_var.get())
        self._render_current_frame(advance_kf=False)

    def _refresh_model_status(self):
        name = os.path.basename(self._model_path)
        if self._model:
            text = f"Model: {self._model_type} ({name})  [{self._device}]"
            color = "#005500"
        else:
            text = f"Model NOT loaded: {self._model_type} ({name})"
            color = "#aa0000"
        self._model_status_var.set(text)
        self._model_status_lbl.config(fg=color)

    # ── Inference ────────────────────────────────────────────────────────

    def _infer(self, image: np.ndarray) -> tuple[list[tuple[float, float, float]], float]:
        if self._model is None:
            return [], 0.0

        tensor = self._transform(image=image)["image"]
        tensor = tensor.unsqueeze(0).to(self._device, dtype=torch.float32)

        if self._device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = self._model(tensor)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

        heatmap = torch.sigmoid(pred[0, 1]).cpu().numpy()
        threshold = self._threshold_var.get()
        nms_radius = int(self._nms_var.get())
        peaks = find_peaks(heatmap, threshold, nms_radius)
        return peaks, infer_ms

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        ctrl = tk.Frame(self, pady=6, padx=8)
        ctrl.pack(fill=tk.X)

        tk.Label(ctrl, text="Model:").pack(side=tk.LEFT)
        self._model_var = tk.StringVar(value=self._model_type)
        model_cb = ttk.Combobox(ctrl, textvariable=self._model_var,
                                values=list(MODEL_REGISTRY), width=12, state="readonly")
        model_cb.pack(side=tk.LEFT, padx=(2, 8))
        model_cb.bind("<<ComboboxSelected>>", self._on_model_change)

        tk.Label(ctrl, text="Method:").pack(side=tk.LEFT)
        self._method_var = tk.StringVar(value=self._arrow.method)
        method_cb = ttk.Combobox(ctrl, textvariable=self._method_var,
                                 values=list(_METHOD_FACTORIES), width=28, state="readonly")
        method_cb.pack(side=tk.LEFT, padx=(2, 8))
        method_cb.bind("<<ComboboxSelected>>", self._on_method_change)

        self._open_btn = tk.Button(
            ctrl, text="Open Video...", command=self._open_video)
        self._open_btn.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(ctrl, text="<-", width=3,
                  command=lambda: self._step(-1)).pack(side=tk.LEFT)
        tk.Button(ctrl, text="->", width=3,
                  command=lambda: self._step(+1)).pack(side=tk.LEFT, padx=(0, 8))

        self._play_btn = tk.Button(ctrl, text="Play", width=6,
                                   command=self._play, state=tk.DISABLED)
        self._play_btn.pack(side=tk.LEFT, padx=(0, 2))
        self._pause_btn = tk.Button(ctrl, text="Pause", width=6,
                                    command=self._pause, state=tk.DISABLED)
        self._pause_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._frame_lbl = tk.Label(ctrl, text="— / —", width=14, anchor="w")
        self._frame_lbl.pack(side=tk.LEFT)

        tk.Label(ctrl, text="  <- -> : step one frame",
                 fg="gray").pack(side=tk.RIGHT)

        # ── Row 2: parameters ──────────────────────────────────────────────
        param = tk.Frame(self, pady=4, padx=8)
        param.pack(fill=tk.X)

        tk.Label(param, text="Threshold:").pack(side=tk.LEFT)
        self._threshold_var = tk.DoubleVar(value=0.5)
        ttk.Scale(param, from_=0.05, to=0.95, orient=tk.HORIZONTAL,
                  variable=self._threshold_var, length=120,
                  command=lambda _: self._on_param_change()
                  ).pack(side=tk.LEFT, padx=(4, 2))
        self._thr_lbl = tk.Label(param, text="0.50", width=5)
        self._thr_lbl.pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(param, text="NMS radius:").pack(side=tk.LEFT)
        self._nms_var = tk.DoubleVar(value=20)
        ttk.Scale(param, from_=5, to=50, orient=tk.HORIZONTAL,
                  variable=self._nms_var, length=120,
                  command=lambda _: self._on_param_change()
                  ).pack(side=tk.LEFT, padx=(4, 2))
        self._nms_lbl = tk.Label(param, text=" 20px", width=5)
        self._nms_lbl.pack(side=tk.LEFT, padx=(0, 12))

        self._model_status_var = tk.StringVar()
        self._model_status_lbl = tk.Label(
            param, textvariable=self._model_status_var,
            font=("Monospace", 9), width=42, anchor="w",
        )
        self._model_status_lbl.pack(side=tk.LEFT)
        self._refresh_model_status()

        # ── Row 3: video panel ─────────────────────────────────────────────
        panel = tk.LabelFrame(
            self,
            text="Video  [ green arrow: >=1 tip detected  pink arrow: 0 tips  "
                 "blue o: raw tip candidates  red polygon: >3 tips (n-gon, capped at pentagon)  "
                 "red X: 2 tips, >90° apart ]",
            padx=4, pady=4,
        )
        panel.pack(padx=8, pady=4)
        self._lbl_video = tk.Label(
            panel, width=PANEL_W, height=PANEL_H, bg="#1a1a1a")
        self._lbl_video.pack()

        # A Label's width/height are character/line counts until an image is
        # actually assigned -- without this placeholder, the empty label (no
        # video opened yet) would blow up to hundreds of "characters" wide
        # instead of PANEL_W pixels, and the whole window would balloon past
        # the screen size.
        self._photo = ImageTk.PhotoImage(
            Image.new("RGB", (PANEL_W, PANEL_H), (26, 26, 26)))
        self._lbl_video.config(image=self._photo)

        # ── Row 4: per-frame info ───────────────────────────────────────────
        info_frame = tk.Frame(self, padx=8, pady=2)
        info_frame.pack(fill=tk.BOTH, expand=True)
        self._info_text = tk.Text(
            info_frame, height=4, wrap=tk.WORD, font=("Monospace", 9),
            relief=tk.FLAT, bg=self.cget("bg"), state=tk.DISABLED,
        )
        self._info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._set_info("Open a video file to begin.")

        # ── Row 5: seek bar ─────────────────────────────────────────────────
        seek_frame = tk.Frame(self, padx=8, pady=2)
        seek_frame.pack(fill=tk.X)
        tk.Label(seek_frame, text="0", width=6, anchor="e").pack(side=tk.LEFT)
        self._seek_var = tk.DoubleVar(value=0)
        self._seekbar = ttk.Scale(seek_frame, from_=0, to=1,
                                  orient=tk.HORIZONTAL, variable=self._seek_var,
                                  command=self._on_seek, state=tk.DISABLED)
        self._seekbar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self._seek_end_lbl = tk.Label(
            seek_frame, text="—", width=6, anchor="w")
        self._seek_end_lbl.pack(side=tk.LEFT)
        self._seek_after_id: str | None = None

    def _set_info(self, text: str):
        self._info_text.config(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)
        self._info_text.insert(tk.END, text)
        self._info_text.config(state=tk.DISABLED)

    # ── Parameter handlers ───────────────────────────────────────────────

    def _on_param_change(self):
        self._thr_lbl.config(text=f"{self._threshold_var.get():.2f}")
        self._nms_lbl.config(text=f"{int(self._nms_var.get()):>3}px")
        if not self._playing:
            self._render_current_frame(advance_kf=False)

    # ── Video loading ────────────────────────────────────────────────────

    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Open Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("All files", "*.*")],
        )
        if not path:
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror(
                "Load Error", f"Could not open video:\n{path}")
            return

        self._pause()
        self._cap = cap
        self._video_path = path
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self._fps <= 0:
            self._fps = 30.0
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._arrow.reset()
        self._seekbar.config(
            state=tk.NORMAL, to=max(1, self._total_frames - 1))
        self._seek_end_lbl.config(text=str(max(0, self._total_frames - 1)))
        self._play_btn.config(state=tk.NORMAL)

        self._goto_frame(0)

    def _current_frame_index(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    # ── Frame reading / rendering ────────────────────────────────────────

    def _read_frame_at(self, index: int) -> np.ndarray | None:
        if self._cap is None:
            return None
        index = max(0, min(index, max(0, self._total_frames - 1)))
        if index != self._current_frame_index():
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame_bgr = self._cap.read()
        if not ok:
            return None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[1] != FRAME_W or frame_rgb.shape[0] != FRAME_H:
            frame_rgb = cv2.resize(frame_rgb, (FRAME_W, FRAME_H),
                                   interpolation=cv2.INTER_LINEAR)
        return frame_rgb

    def _goto_frame(self, index: int):
        """Manual scrub to an arbitrary frame; processed once, KF still updates."""
        frame = self._read_frame_at(index)
        if frame is None:
            return
        self._last_frame = frame
        self._update_frame_label()
        self._process_and_show(frame)

    def _step(self, delta: int):
        if self._cap is None:
            return
        self._pause()
        # current_frame_index() is the next unread frame (i.e. last_shown + 1),
        # so last_shown + delta == current_frame_index() + delta - 1 for either sign.
        self._goto_frame(self._current_frame_index() + delta - 1)

    def _render_current_frame(self, *, advance_kf: bool):
        """Re-render the last frame (e.g. after a threshold/NMS slider change)
        without consuming a new video frame or advancing the arrow state."""
        if getattr(self, "_last_frame", None) is None:
            return
        self._process_and_show(self._last_frame, advance_kf=advance_kf)

    def _process_and_show(self, frame: np.ndarray, *, advance_kf: bool = True):
        peaks, infer_ms = self._infer(frame)

        if advance_kf:
            smoothed = self._arrow.update(peaks)
        else:
            smoothed = self._arrow.vector

        # Arrow color reflects raw tip *count* only (0 => pink), independent of
        # whether ArrowState filtered this frame out as a likely false
        # positive -- that's what the overflow polygon / decay rate are for.
        color = _COLOR_NOT_DETECTED if len(peaks) == 0 else _COLOR_DETECTED

        display = _draw_raw_markers(frame.copy(), peaks)
        display = _draw_arrow(display, smoothed, color)

        if len(peaks) > _TIP_OVERFLOW_THRESHOLD:
            display = _draw_tip_overflow_polygon(display, len(peaks))
        elif self._arrow.two_tip_conflict:
            display = _draw_two_tip_conflict_x(display)

        self._photo = _to_photo(display)
        self._lbl_video.config(image=self._photo)

        measurement = self._arrow.last_measurement
        self._set_info(
            f"inference: {infer_ms:.1f} ms\n"
            f"{self._arrow.reason}\n"
            f"raw measurement: ({measurement[0]:+.1f}, {measurement[1]:+.1f})\n"
            f"smoothed arrow: len={self._arrow.length:.1f}px  angle={self._arrow.angle_deg:+.1f}deg"
        )

    def _update_frame_label(self):
        idx = self._current_frame_index() - 1  # after read(), pos already advanced
        idx = max(0, idx)
        self._frame_lbl.config(
            text=f"{idx} / {max(0, self._total_frames - 1)}")
        self._seek_var.set(idx)

    # ── Playback ─────────────────────────────────────────────────────────

    def _play(self):
        if self._cap is None or self._playing:
            return
        self._playing = True
        self._play_btn.config(state=tk.DISABLED)
        self._pause_btn.config(state=tk.NORMAL)
        self._play_start_wall = time.perf_counter()
        self._play_start_frame = self._current_frame_index()
        self._schedule_play_tick()

    def _pause(self):
        self._playing = False
        if self._play_after_id is not None:
            self.after_cancel(self._play_after_id)
            self._play_after_id = None
        self._play_btn.config(state=tk.NORMAL if self._cap else tk.DISABLED)
        self._pause_btn.config(state=tk.DISABLED)

    def _schedule_play_tick(self):
        self._play_after_id = self.after(1, self._play_tick)

    def _play_tick(self):
        if not self._playing or self._cap is None:
            return

        elapsed = time.perf_counter() - self._play_start_wall
        target = self._play_start_frame + int(elapsed * self._fps)

        if target >= self._total_frames > 0:
            self._pause()
            return

        frame = self._read_frame_at(max(target, self._current_frame_index()))
        if frame is None:
            self._pause()
            return

        self._last_frame = frame
        self._update_frame_label()
        self._process_and_show(frame)

        self._schedule_play_tick()

    def _on_seek(self, _=None):
        if self._seek_after_id is not None:
            self.after_cancel(self._seek_after_id)
        self._seek_after_id = self.after(150, self._apply_seek)

    def _apply_seek(self):
        self._seek_after_id = None
        if self._cap is None:
            return
        self._pause()
        idx = int(round(self._seek_var.get()))
        self._goto_frame(idx)

    def _on_close(self):
        self._pause()
        if self._cap is not None:
            self._cap.release()
        self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive GUI for real-time surgical tool tip tracking in video"
    )
    parser.add_argument("--model-type", default="monai",
                        choices=list(MODEL_REGISTRY),
                        help="Model architecture to load (default: monai)")
    parser.add_argument("--model", default=None,
                        help="Path to trained model weights "
                             "(default: data/models/<model-type>/best.pt)")
    args = parser.parse_args()

    model_path = args.model or _default_model_path(args.model_type)

    app = TooltipTrackerApp(model_type=args.model_type, model_path=model_path)
    app.mainloop()


if __name__ == "__main__":
    main()
