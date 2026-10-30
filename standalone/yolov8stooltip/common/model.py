"""YOLOv8s with a YOLOv8-pose head: each detection is a tool box *and* its tip.

The backbone and neck are the YOLOv8s clone's (baseline/yolov8sclone), layer
for layer. Only the head differs. The clone detects `tool` boxes and `tip`
boxes as two unrelated classes, so which tip belongs to which tool has to be
guessed afterwards, and close or overlapping tools get the wrong tip. Here the
head follows YOLOv8-pose instead:

    box branch       4 x 16 DFL logits  -> the tool box
    class branch     1 logit            -> `tool` score
    keypoint branch  2 values           -> the tip (x, y) of *that* tool

All three come from the same anchor point, so a tip is tied to its box by
construction and no matching is needed at inference. The keypoint has no
visibility channel: every annotated tool has a tip, so there is nothing to
learn it from.

     0-9   backbone (CSPDarknet + C2f + SPPF)           as YOLOv8s
    10-21  neck (PAN-FPN)                               as YOLOv8s
    22     Pose(nc=1, kpt_shape=(1, 2))

Keypoints are decoded as in YOLOv8-pose: the raw value r of an anchor point
at grid cell centre a and stride s gives (2r + a - 0.5) * s pixels, so the tip
can lie anywhere, not only inside the anchor's cell.
"""

import math

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

CLASS_NAMES = ("tool",)
# One keypoint, the tip, as (x, y) with no visibility channel.
KPT_SHAPE = (1, 2)
STRIDES = (8, 16, 32)
REG_MAX = 16                 # DFL bins per box side


def autopad(k, p=None, d=1):
    """'same' padding. `k` may be an int or a per-axis pair -- C2f's
    bottlenecks pass (3, 3)."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is not None:
        return p
    return k // 2 if isinstance(k, int) else [x // 2 for x in k]


class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse(self) -> None:
        """Fold `bn` into `conv`. Inference only, and not reversible.

        In eval mode BatchNorm is a fixed affine map, so folding it into the
        preceding convolution's weight and bias computes the same thing with
        one kernel instead of two. Doing nothing when `bn` is already gone
        keeps this idempotent.
        """
        if isinstance(self.bn, nn.BatchNorm2d):
            self.conv = fuse_conv_bn_eval(self.conv, self.bn)
            self.bn = nn.Identity()


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP block with two convolutions and `n` bottlenecks, YOLOv8's C2f.

    Unlike YOLOv5's C3, every bottleneck's output is kept and concatenated, so
    the fusion convolution sees `2 + n` branches instead of two.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
                               for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial pyramid pooling, fast variant."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class DFL(nn.Module):
    """Distribution Focal Loss integral: turn `reg_max` logits per box side
    into one expected distance, with a fixed (non-learned) 0..reg_max-1 kernel."""

    def __init__(self, c1=REG_MAX):
        super().__init__()
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(
            torch.arange(c1, dtype=torch.float32).view(1, c1, 1, 1))

    def forward(self, x):
        b, _, a = x.shape                      # (batch, 4 * reg_max, anchors)
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect(nn.Module):
    """Anchor-free decoupled head: a DFL box branch and a class branch."""

    def __init__(self, nc: int, ch: tuple[int, ...]):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = REG_MAX
        self.no = nc + self.reg_max * 4
        self.stride = torch.tensor(STRIDES, dtype=torch.float32)

        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for x in ch)
        self.dfl = DFL(self.reg_max)
        self._bias_init()

    def _bias_init(self):
        """Start with a low class prior; the anchor-free head has far more
        negatives than positives on the first batches."""
        for box, cls, stride in zip(self.cv2, self.cv3, self.stride):
            box[-1].bias.data[:] = 1.0
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / float(stride)) ** 2)

    def forward(self, features):
        return [torch.cat((self.cv2[i](f), self.cv3[i](f)), 1) for i, f in enumerate(features)]


class Pose(Detect):
    """YOLOv8-pose head: Detect plus a keypoint branch on the same anchors."""

    def __init__(self, nc: int, ch: tuple[int, ...], kpt_shape: tuple[int, int] = KPT_SHAPE):
        super().__init__(nc, ch)
        self.kpt_shape = tuple(kpt_shape)
        self.nk = self.kpt_shape[0] * self.kpt_shape[1]
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1))
            for x in ch)

    def forward(self, features):
        """(Detect's per-level outputs, raw keypoints (B, nk, n_anchors))."""
        batch = features[0].shape[0]
        keypoints = torch.cat([self.cv4[i](f).view(batch, self.nk, -1)
                               for i, f in enumerate(features)], -1)
        return super().forward(features), keypoints


class YOLOv8Tooltip(nn.Module):
    """YOLOv8 with the published depth/width scaling and a pose head. `s` is the default."""

    SCALES = {                       # depth, width, max_channels
        "n": (0.33, 0.25, 1024),
        "s": (0.33, 0.50, 1024),
        "m": (0.67, 0.75, 768),
        "l": (1.00, 1.00, 512),
        "x": (1.00, 1.25, 512),
    }

    def __init__(self, num_classes: int = len(CLASS_NAMES), scale: str = "s",
                 kpt_shape: tuple[int, int] = KPT_SHAPE):
        super().__init__()
        if scale not in self.SCALES:
            raise ValueError(f"scale must be one of {tuple(self.SCALES)}")
        depth, width, max_channels = self.SCALES[scale]
        self.scale = scale
        self.num_classes = num_classes
        self.strides = STRIDES
        self.reg_max = REG_MAX
        self.num_outputs = num_classes + REG_MAX * 4

        def ch(c):                    # channel count after width scaling
            return int(min(c, max_channels) * width)

        def n(repeats):               # bottleneck repeats after depth scaling
            return max(round(repeats * depth), 1)

        c1, c2, c3, c4, c5 = ch(64), ch(128), ch(256), ch(512), ch(1024)

        # backbone
        self.layer0 = Conv(3, c1, 3, 2)
        self.layer1 = Conv(c1, c2, 3, 2)
        self.layer2 = C2f(c2, c2, n(3), shortcut=True)
        self.layer3 = Conv(c2, c3, 3, 2)
        self.layer4 = C2f(c3, c3, n(6), shortcut=True)
        self.layer5 = Conv(c3, c4, 3, 2)
        self.layer6 = C2f(c4, c4, n(6), shortcut=True)
        self.layer7 = Conv(c4, c5, 3, 2)
        self.layer8 = C2f(c5, c5, n(3), shortcut=True)
        self.layer9 = SPPF(c5, c5, 5)

        # head: top-down then bottom-up (PAN)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.layer12 = C2f(c5 + c4, c4, n(3))
        self.layer15 = C2f(c4 + c3, c3, n(3))
        self.layer16 = Conv(c3, c3, 3, 2)
        self.layer18 = C2f(c3 + c4, c4, n(3))
        self.layer19 = Conv(c4, c4, 3, 2)
        self.layer21 = C2f(c4 + c5, c5, n(3))

        self.kpt_shape = tuple(kpt_shape)
        self.detect = Pose(num_classes, (c3, c4, c5), self.kpt_shape)

    def forward(self, x):
        """Return (one raw tensor per level shaped (B, nc + 4 * reg_max, H, W),
        raw keypoints shaped (B, nk, n_anchors))."""
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        p3 = self.layer4(x)
        x = self.layer5(p3)
        p4 = self.layer6(x)
        x = self.layer7(p4)
        x = self.layer8(x)
        p5 = self.layer9(x)

        x = self.layer12(torch.cat([self.upsample(p5), p4], 1))
        n3 = self.layer15(torch.cat([self.upsample(x), p3], 1))
        n4 = self.layer18(torch.cat([self.layer16(n3), x], 1))
        n5 = self.layer21(torch.cat([self.layer19(n4), p5], 1))
        return self.detect((n3, n4, n5))


def make_anchors(features, strides, grid_cell_offset: float = 0.5):
    """Anchor-point centres and their strides, concatenated over all levels."""
    points, stride_tensor = [], []
    dtype, device = features[0].dtype, features[0].device
    for feature, stride in zip(features, strides):
        _, _, h, w = feature.shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), float(stride), dtype=dtype, device=device))
    return torch.cat(points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh: bool = True, dim: int = -1):
    """Left/top/right/bottom distances from an anchor point -> a box."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points, bbox_xyxy, reg_max: int):
    """Inverse of dist2bbox, clamped into the DFL's representable range."""
    x1y1, x2y2 = bbox_xyxy.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


def decode_keypoints(keypoints: torch.Tensor, anchor_points: torch.Tensor,
                     ndim: int) -> torch.Tensor:
    """Raw keypoints (B, nk, A) -> grid units, as YOLOv8-pose decodes them.

    `anchor_points` is (A, 2) in grid units (cell centres). Multiply by the
    stride for pixels.
    """
    decoded = keypoints.clone()
    decoded[:, 0::ndim] = decoded[:, 0::ndim] * 2.0 + (anchor_points[:, 0] - 0.5)
    decoded[:, 1::ndim] = decoded[:, 1::ndim] * 2.0 + (anchor_points[:, 1] - 0.5)
    return decoded


def decode(outputs, model) -> torch.Tensor:
    """Raw head outputs -> (B, N, 4 + nc + nk): [cx, cy, w, h] in input
    pixels, sigmoid class scores, then each anchor's keypoints in input pixels.
    There is no objectness term to fold in."""
    features, keypoints = outputs
    detect = model.detect
    batch = features[0].shape[0]
    flattened = torch.cat([o.view(batch, detect.no, -1) for o in features], 2)
    box, cls = flattened.split((detect.reg_max * 4, detect.nc), 1)

    anchor_points, stride_tensor = make_anchors(features, model.strides)
    boxes = dist2bbox(detect.dfl(box), anchor_points.transpose(0, 1).unsqueeze(0), xywh=True, dim=1)
    boxes = boxes * stride_tensor.transpose(0, 1)
    points = decode_keypoints(keypoints, anchor_points, detect.kpt_shape[1])
    points = points * stride_tensor.transpose(0, 1)
    return torch.cat([boxes, cls.sigmoid(), points], 1).permute(0, 2, 1).contiguous()


def build(num_classes: int = len(CLASS_NAMES), scale: str = "s",
          kpt_shape: tuple[int, int] = KPT_SHAPE) -> YOLOv8Tooltip:
    return YOLOv8Tooltip(num_classes=num_classes, scale=scale, kpt_shape=kpt_shape)


def fuse(model: nn.Module) -> nn.Module:
    """Fold every `Conv`'s BatchNorm into its convolution. Returns `model`.

    Call it on an eval-mode model that already holds its trained weights: the
    fold reads BatchNorm's running statistics, and it rewrites `state_dict`
    keys (`*.bn.*` disappears, `*.conv.bias` appears). So a fused model can no
    longer load or save a normal checkpoint, which is why this is not part of
    `build` and why `Detector` calls it only after `load_state_dict`.

    Ultralytics fuses the same way when `AutoBackend` loads a checkpoint, so
    this is what the reference runs at inference too.
    """
    if model.training:
        raise RuntimeError("fuse() needs an eval-mode model: BatchNorm's running "
                           "statistics are what gets folded in")
    for module in model.modules():
        if isinstance(module, Conv):
            module.fuse()
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
