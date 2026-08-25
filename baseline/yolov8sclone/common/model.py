"""YOLOv8s, reimplemented in plain PyTorch.

A from-scratch rebuild of the architecture behind
`cesaraha/yolov8s-surgical-instrument-detection-cholec80` (see
baseline/yolov8s), written so that nothing here -- training, evaluation or the
demo -- imports `ultralytics`. Only packages the root project already depends
on are used.

The layout follows the published YOLOv8 configuration exactly, including the
module numbering, so a structural mistake shows up as a parameter-count
mismatch against the reference checkpoint (11.14 M at nc=7):

     0 Conv  3->32   s2      P1/2
     1 Conv  32->64  s2      P2/4
     2 C2f   64      n=1
     3 Conv  64->128 s2      P3/8
     4 C2f   128     n=2
     5 Conv  128->256 s2     P4/16
     6 C2f   256     n=2
     7 Conv  256->512 s2     P5/32
     8 C2f   512     n=1
     9 SPPF  512
    10-15 top-down  (upsample, concat P4, C2f; upsample, concat P3, C2f) -> N3
    16-18 bottom-up (Conv s2, concat, C2f)                               -> N4
    19-21 bottom-up (Conv s2, concat, C2f)                               -> N5
    22 Detect(nc)

What separates this from the CLAD-Net baseline next door, and the reason both
exist: YOLOv8 is **anchor-free**, has **no objectness branch**, regresses each
box side as a **discrete distribution over 16 bins** (DFL) instead of a single
number, and assigns labels **dynamically** (TaskAlignedAssigner, see
common/assigner.py) rather than by anchor-shape matching.

Two classes are detected, both derived from this repository's annotations:

    0  tool   the annotated bounding box
    1  tip    a fixed-size box centred on the annotated tool tip
"""

import math

import torch
import torch.nn as nn

CLASS_NAMES = ("tool", "tip")
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


class YOLOv8(nn.Module):
    """YOLOv8 with the published depth/width scaling. `s` is the default."""

    SCALES = {                       # depth, width, max_channels
        "n": (0.33, 0.25, 1024),
        "s": (0.33, 0.50, 1024),
        "m": (0.67, 0.75, 768),
        "l": (1.00, 1.00, 512),
        "x": (1.00, 1.25, 512),
    }

    def __init__(self, num_classes: int = len(CLASS_NAMES), scale: str = "s"):
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

        self.detect = Detect(num_classes, (c3, c4, c5))

    def forward(self, x):
        """Return one raw tensor per level, shaped (B, nc + 4 * reg_max, H, W)."""
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


def decode(outputs, model) -> torch.Tensor:
    """Raw head outputs -> (B, N, 4 + nc) with [cx, cy, w, h] in input pixels
    and sigmoid class scores. There is no objectness term to fold in."""
    detect = model.detect
    batch = outputs[0].shape[0]
    flattened = torch.cat([o.view(batch, detect.no, -1) for o in outputs], 2)
    box, cls = flattened.split((detect.reg_max * 4, detect.nc), 1)

    anchor_points, stride_tensor = make_anchors(outputs, model.strides)
    boxes = dist2bbox(detect.dfl(box), anchor_points.transpose(0, 1).unsqueeze(0), xywh=True, dim=1)
    boxes = boxes * stride_tensor.transpose(0, 1)
    return torch.cat([boxes, cls.sigmoid()], 1).permute(0, 2, 1).contiguous()


def build(num_classes: int = len(CLASS_NAMES), scale: str = "s") -> YOLOv8:
    return YOLOv8(num_classes=num_classes, scale=scale)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
