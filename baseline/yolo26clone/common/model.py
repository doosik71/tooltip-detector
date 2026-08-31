"""YOLO26, reimplemented in plain PyTorch.

A from-scratch rebuild of the architecture Ultralytics ships as YOLO26 (see
baseline/yolo26, which runs the package itself), written so that nothing here
-- training, evaluation or the demo -- imports `ultralytics`. Only packages the
root project already depends on are used.

The module list follows the published `yolo26.yaml` exactly, index for index,
and every submodule keeps the reference implementation's attribute names. That
is deliberate: it makes the reference checkpoint's `state_dict` load into this
model with **no key renaming at all**, so `scripts/verify-clone.py` can prove
the rebuild is right by loading those weights and comparing outputs, not just
by counting parameters.

     0 Conv    3->32   s2                      P1/2
     1 Conv    32->64  s2                      P2/4
     2 C3k2    64->128   n=1 c3k=False e=0.25
     3 Conv    128->128 s2                     P3/8
     4 C3k2    128->256  n=1 c3k=False e=0.25
     5 Conv    256->256 s2                     P4/16
     6 C3k2    256->256  n=1 c3k=True
     7 Conv    256->512 s2                     P5/32
     8 C3k2    512->512  n=1 c3k=True
     9 SPPF    512->512  k=5 n=3 shortcut
    10 C2PSA   512->512  n=1
    11-16 top-down  (upsample, concat P4, C3k2; upsample, concat P3, C3k2) -> N3
    17-19 bottom-up (Conv s2, concat, C3k2)                                -> N4
    20-22 bottom-up (Conv s2, concat, C3k2 with attention)                 -> N5
    23 Detect(nc), end-to-end

Channels are for `--scale s` at nc=2, which is 9,949,412 parameters.

What separates this from the YOLOv8s clone next door -- the reason both exist:

  end-to-end head   Detect carries two parallel branches. The one-to-many
                    branch is YOLOv8's, trained with topk=10 assignment; the
                    one-to-one branch is trained to leave exactly one box per
                    object, and it is the only one used at inference. There is
                    no NMS anywhere, so `postprocess` is a top-k selection.
  no DFL            `reg_max` is 1, so each box side is one regressed number
                    rather than a distribution over 16 bins, and the third
                    loss term is an L1 instead of a distribution loss.
  C3k2 / C2PSA      the CSP block can nest a C3k block or an attention block,
                    and the backbone ends with a position-sensitive attention
                    stage rather than plain convolutions.
  DWConv classifier the class branch starts with a depthwise 3x3 per level
                    instead of two full 3x3 convolutions.

Two classes are detected, both derived from this repository's annotations:

    0  tool   the annotated bounding box
    1  tip    a fixed-size box centred on the annotated tool tip
"""

import copy
import math

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

CLASS_NAMES = ("tool", "tip")
STRIDES = (8, 16, 32)
REG_MAX = 1                  # YOLO26 regresses distances directly: no DFL bins
MAX_DET = 300                # boxes kept per image by the end-to-end head

# BatchNorm settings the reference sets on every module after building it, and
# which are *not* PyTorch's defaults (1e-5 and 0.1). They change what the
# network computes, not just how it trains: at eps 1e-5 this clone's output
# drifts from the reference's by hundreds of units on the first layer alone.
BN_EPS = 1e-3
BN_MOMENTUM = 0.03


def autopad(k, p=None, d=1):
    """'same' padding. `k` may be an int or a per-axis pair."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is not None:
        return p
    return k // 2 if isinstance(k, int) else [x // 2 for x in k]


def make_divisible(x: float, divisor: int = 8) -> int:
    """Round a width-scaled channel count up to a multiple of `divisor`."""
    return math.ceil(x / divisor) * divisor


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


class DWConv(Conv):
    """Depthwise convolution: one group per input channel."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


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
    """CSP block with two convolutions and `n` blocks, YOLOv8's C2f.

    Every block's output is kept and concatenated, so the fusion convolution
    sees `2 + n` branches. C3k2 below only changes what the blocks are.
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


class C3(nn.Module):
    """CSP block with three convolutions, YOLOv5's C3. Only used inside C3k."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0)
                                 for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """A C3 block with a configurable kernel, used as one C3k2 branch."""


class Attention(nn.Module):
    """Multi-head self-attention over the feature map, plus a depthwise
    positional encoding added to the values.

    Queries, keys and values come out of one 1x1 convolution. The key
    dimension is `attn_ratio` of the head dimension, so the projection is
    cheaper than a square attention would be.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        h = dim + self.key_dim * num_heads * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        b, c, height, width = x.shape
        n = height * width
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, height, width) + self.pe(v.reshape(b, c, height, width))
        return self.proj(x)


class PSABlock(nn.Module):
    """Position-sensitive attention block: attention then a 1x1 feed-forward,
    both with residual connections."""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C2PSA(nn.Module):
    """Split the channels, run PSA blocks on one half, concatenate back.

    This is the stage that ends YOLO26's backbone. YOLOv8 has plain
    convolutions there.
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
                                 for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


class C3k2(C2f):
    """C2f whose blocks can be a plain bottleneck, a C3k, or a
    bottleneck followed by a PSA block."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(Bottleneck(self.c, self.c, shortcut, g),
                          PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)))
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n))


class SPPF(nn.Module):
    """Spatial pyramid pooling, fast variant.

    YOLO26's version differs from YOLOv8's in three details that all show up
    in the weights: the entry convolution has no activation, the number of
    pooling steps is an argument (3 here), and the block can be residual.
    """

    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class Detect(nn.Module):
    """End-to-end anchor-free head with two parallel branches.

    `cv2`/`cv3` are the one-to-many branch, assigned exactly as YOLOv8's head
    is. `one2one_cv2`/`one2one_cv3` are a structural copy of them, trained
    with a stricter assignment so that each object ends up with a single box.
    Only the one-to-one branch is read at inference, which is what removes the
    need for NMS.

    The one-to-one branch sees detached features. Its gradients train the head
    only, so the shared backbone is shaped by the one-to-many branch alone.
    """

    def __init__(self, nc: int, ch: tuple[int, ...], reg_max: int = REG_MAX):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.tensor(STRIDES, dtype=torch.float32)
        self.max_det = MAX_DET
        # Skip the one-to-many branch, which detection does not read. Off by
        # default because both the loss and `verify-clone` need that branch;
        # only `Detector` turns it on. See `forward`.
        self.one2one_only = False

        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1))
            for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                          nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                          nn.Conv2d(c3, nc, 1))
            for x in ch)
        # A structural copy of both branches. The weights are re-initialised
        # right after, so only the shapes carry over.
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
        self._bias_init()

    def _bias_init(self):
        """Start with a low class prior: an anchor-free head sees far more
        negatives than positives on the first batches. The box branch starts
        at 2.0, a positive distance in stride units."""
        for boxes, classes in ((self.cv2, self.cv3), (self.one2one_cv2, self.one2one_cv3)):
            for box, cls, stride in zip(boxes, classes, self.stride):
                box[-1].bias.data[:] = 2.0
                cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / float(stride)) ** 2)

    def _branch(self, features, box_head, cls_head) -> dict:
        batch = features[0].shape[0]
        return {
            "boxes": torch.cat([box_head[i](f).view(batch, 4 * self.reg_max, -1)
                                for i, f in enumerate(features)], dim=-1),
            "scores": torch.cat([cls_head[i](f).view(batch, self.nc, -1)
                                 for i, f in enumerate(features)], dim=-1),
            "feats": features,
        }

    def forward(self, features) -> dict:
        """Both branches, each as {boxes, scores, feats}.

        The same dict shape is returned in training and in evaluation; turning
        it into boxes is `decode`'s job, so nothing about the head depends on
        which mode it is in.

        `one2one_only` is the one exception. Detection reads `one2one` and
        throws `one2many` away, so computing the discarded branch costs about
        15 % of the frame time for nothing. The flag drops it. It cannot be
        inferred from `self.training`: `verify-clone` compares `one2many`
        against the reference on a model in eval mode, and the loss needs the
        branch in every validation pass. So the flag is opt-in, and training
        mode ignores it rather than losing a branch the loss requires.
        """
        if self.one2one_only and not self.training:
            return {"one2one": self._branch(features, self.one2one_cv2, self.one2one_cv3)}
        detached = [f.detach() for f in features] if self.training else features
        return {
            "one2many": self._branch(features, self.cv2, self.cv3),
            "one2one": self._branch(detached, self.one2one_cv2, self.one2one_cv3),
        }


class YOLO26(nn.Module):
    """YOLO26 with the published depth/width scaling. `s` is the default.

    The layers live in an `nn.ModuleList` called `model`, indexed exactly as
    the reference configuration numbers them, so the parameter names this
    module reports are the reference's parameter names.
    """

    SCALES = {                       # depth, width, max_channels
        "n": (0.50, 0.25, 1024),
        "s": (0.50, 0.50, 1024),
        "m": (0.50, 1.00, 512),
        "l": (1.00, 1.00, 512),
        "x": (1.00, 1.50, 512),
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
            return make_divisible(min(c, max_channels) * width, 8)

        def n(repeats):               # block repeats after depth scaling
            return max(round(repeats * depth), 1) if repeats > 1 else repeats

        c1, c2, c3, c4, c5 = ch(64), ch(128), ch(256), ch(512), ch(1024)
        upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.model = nn.ModuleList([
            Conv(3, c1, 3, 2),                                      # 0  P1/2
            Conv(c1, c2, 3, 2),                                     # 1  P2/4
            C3k2(c2, c3, n(2), c3k=False, e=0.25),                  # 2
            Conv(c3, c3, 3, 2),                                     # 3  P3/8
            C3k2(c3, c4, n(2), c3k=False, e=0.25),                  # 4
            Conv(c4, c4, 3, 2),                                     # 5  P4/16
            C3k2(c4, c4, n(2), c3k=True),                           # 6
            Conv(c4, c5, 3, 2),                                     # 7  P5/32
            C3k2(c5, c5, n(2), c3k=True),                           # 8
            SPPF(c5, c5, 5, 3, True),                               # 9
            C2PSA(c5, c5, n(2)),                                    # 10
            upsample,                                               # 11
            nn.Identity(),                                          # 12 concat
            C3k2(c5 + c4, c4, n(2), c3k=True),                      # 13
            upsample,                                               # 14
            nn.Identity(),                                          # 15 concat
            C3k2(c4 + c4, c3, n(2), c3k=True),                      # 16 -> N3
            Conv(c3, c3, 3, 2),                                     # 17
            nn.Identity(),                                          # 18 concat
            C3k2(c3 + c4, c4, n(2), c3k=True),                      # 19 -> N4
            Conv(c4, c4, 3, 2),                                     # 20
            nn.Identity(),                                          # 21 concat
            C3k2(c4 + c5, c5, 1, c3k=True, e=0.5, attn=True),       # 22 -> N5
            Detect(num_classes, (c3, c4, c5)),                      # 23
        ])
        self._set_batchnorm_defaults()

    def _set_batchnorm_defaults(self):
        """Apply the reference's BatchNorm epsilon and momentum everywhere."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = BN_EPS
                module.momentum = BN_MOMENTUM

    @property
    def detect(self) -> Detect:
        """The head. A property, not an attribute: registering it twice would
        duplicate every head parameter in `state_dict`."""
        return self.model[-1]

    def forward(self, x) -> dict:
        """Return {one2many, one2one}, each {boxes, scores, feats}."""
        layer = self.model
        x = layer[1](layer[0](x))
        x = layer[2](x)
        x = layer[3](x)
        x = layer[4](x)
        p3 = x
        x = layer[6](layer[5](x))
        p4 = x
        x = layer[8](layer[7](x))
        p5 = layer[10](layer[9](x))

        x = layer[13](torch.cat([layer[11](p5), p4], 1))
        n3 = layer[16](torch.cat([layer[14](x), p3], 1))
        n4 = layer[19](torch.cat([layer[17](n3), x], 1))
        n5 = layer[22](torch.cat([layer[20](n4), p5], 1))
        return layer[23]((n3, n4, n5))


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


def dist2bbox(distance, anchor_points, xywh: bool = False, dim: int = -1):
    """Left/top/right/bottom distances from an anchor point -> a box."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points, bbox_xyxy):
    """Inverse of dist2bbox. Nothing is clamped: without DFL bins there is no
    representable range to clamp into."""
    x1y1, x2y2 = bbox_xyxy.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)


def decode(preds: dict, strides=STRIDES) -> torch.Tensor:
    """One branch's raw output -> (B, N, 4 + nc), boxes as xyxy in input
    pixels and sigmoid class scores.

    There is no objectness term to fold in, and -- unlike the YOLOv8 clone --
    no DFL integral either: `boxes` already holds the four distances.
    """
    anchor_points, stride_tensor = make_anchors(preds["feats"], strides)
    boxes = dist2bbox(preds["boxes"], anchor_points.transpose(0, 1).unsqueeze(0), dim=1)
    boxes = boxes * stride_tensor.transpose(0, 1)
    return torch.cat([boxes, preds["scores"].sigmoid()], 1).permute(0, 2, 1).contiguous()


def postprocess(decoded: torch.Tensor, max_det: int = MAX_DET) -> torch.Tensor:
    """Top-k selection over (B, N, 4 + nc) -> (B, k, 6) [x1,y1,x2,y2,score,cls].

    This is what replaces NMS. The one-to-one branch is trained to leave one
    box per object, so the only thing left to do is keep the best `max_det`
    (anchor, class) pairs: first the anchors with the highest score of any
    class, then the best class-scores among those.
    """
    boxes, scores = decoded.split([4, decoded.shape[-1] - 4], dim=-1)
    n_anchors, nc = scores.shape[1:]
    k = min(max_det, n_anchors)

    anchor_index = scores.amax(dim=-1).topk(k, dim=1).indices
    scores = scores.gather(1, anchor_index[..., None].expand(-1, -1, nc))
    top_scores, flat_index = scores.flatten(1).topk(k, dim=1)
    classes = (flat_index % nc)[..., None].float()
    picked = anchor_index.gather(1, flat_index // nc)
    boxes = boxes.gather(1, picked[..., None].expand(-1, -1, 4))
    return torch.cat([boxes, top_scores[..., None], classes], dim=-1)


def build(num_classes: int = len(CLASS_NAMES), scale: str = "s") -> YOLO26:
    return YOLO26(num_classes=num_classes, scale=scale)


def fuse(model: nn.Module) -> nn.Module:
    """Fold every `Conv`'s BatchNorm into its convolution. Returns `model`.

    Call it on an eval-mode model that already holds its trained weights: the
    fold reads BatchNorm's running statistics, and it rewrites `state_dict`
    keys (`*.bn.*` disappears, `*.conv.bias` appears). So a fused model can no
    longer load or save a normal checkpoint, which is why this is not part of
    `build` and why `Detector` calls it only after `load_state_dict`.

    The reference does the same thing at inference (`AutoBackend` fuses on
    load), so this is not a departure from it. Detections are unchanged: in
    eval mode the two forms are the same function.
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
