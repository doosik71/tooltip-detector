"""The three modules CLAD-Net contributes, implemented from the paper.

Zhao et al., "CLAD-Net: cross-layer aggregation attention network for
real-time endoscopic instrument detection", Health Inf Sci Syst 11:58 (2023).

The authors released no code, so every module here is reconstructed from the
text and from Figs. 2-3. Where the text and the figures disagree, or where a
hyper-parameter is simply not stated, the choice is marked with a `PAPER:`
comment naming what was decided and why. Those comments are the checklist to
revisit if a reproduction attempt falls short of the published numbers.
"""

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
import torch.nn.functional as F


def autopad(k: int, p: int | None = None) -> int:
    return k // 2 if p is None else p


class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU, the standard YOLO convolution block."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1,
                 p: int | None = None, g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
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


class DWConv(nn.Module):
    """Depthwise separable convolution: depthwise 3x3 then pointwise 1x1.

    The paper uses this on the C3/C4/C5 lateral connections -- "we combine
    depthwise separable convolution in the cross-layer aggregation attention
    module to extract deeper features".
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        self.dw = Conv(c1, c1, k, s, g=c1)
        self.pw = Conv(c1, c2, 1, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class AAB(nn.Module):
    """Adaptive Attention Branching (paper Fig. 2, upper half).

    Pools the input to four different sizes, brings each back to the input
    resolution, and turns the concatenation into one spatial+channel gate that
    is multiplied back onto the input.
    """

    # PAPER: "4 contextual features of different scales" -- the actual output
    # sizes are never given. These are the PSPNet pyramid sizes, the usual
    # choice for exactly this construction.
    DEFAULT_POOL_SIZES = (1, 2, 3, 6)

    def __init__(self, c: int, pool_sizes: tuple[int, ...] = DEFAULT_POOL_SIZES,
                 reduction: int = 4):
        super().__init__()
        self.pool_sizes = pool_sizes
        cr = max(c // reduction, 8)
        # One 1x1 per pyramid level, "adjusts the number of channels". Kept as
        # a bare convolution: the 1x1 pyramid level pools to a single pixel,
        # where a BatchNorm has nothing to normalise over.
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, cr, 1, bias=True) for _ in pool_sizes])
        # PAPER: the text orders these "Conv1x1 -> ReLU -> BatchNorm -> Conv3x3
        # -> Sigmoid" but Fig. 2 draws "Conv1x1 -> BatchNorm -> ReLU ->
        # Conv3x3 -> Sigmoid". The figure's order is followed here.
        self.fuse = nn.Sequential(
            nn.Conv2d(cr * len(pool_sizes), cr, 1, bias=False),
            nn.BatchNorm2d(cr),
            nn.ReLU(inplace=True),
            nn.Conv2d(cr, c, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        pooled = [
            F.interpolate(lateral(F.adaptive_avg_pool2d(x, size)), size=(h, w),
                          mode="bilinear", align_corners=False)
            for size, lateral in zip(self.pool_sizes, self.laterals)
        ]
        f_a = self.fuse(torch.cat(pooled, 1))
        return f_a * x


class MSAB(nn.Module):
    """Multi-Scale Attention Branching (paper Fig. 2, lower half).

    Two channel-attention branches summed before one sigmoid:
      GCA  global average pool first -> one weight per channel  (C x 1 x 1)
      LCA  no pooling                -> one weight per channel *per pixel*
    """

    def __init__(self, c: int, reduction: int = 16):
        super().__init__()
        # PAPER: "reduce the number of parameters and complexity through 1x1
        # convolution" without giving a ratio. 16 is the SE-block default.
        cr = max(c // reduction, 8)
        self.gca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
            nn.ReLU(inplace=True),
            nn.Conv2d(cr, c, 1, bias=True),
        )
        self.lca = nn.Sequential(
            nn.Conv2d(c, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
            nn.ReLU(inplace=True),
            nn.Conv2d(cr, c, 1, bias=True),
        )

    def forward(self, x):
        f_m = torch.sigmoid(self.gca(x) + self.lca(x))   # broadcast C x 1 x 1 over C x H x W
        return f_m * x


class CAM(nn.Module):
    """Composite Attention Mechanism -- paper Eq. (1), Fig. 2.

        F_y = Conv1x1( Concat[ AAB(F_x), MSAB(F_x) ] )

    `c_out` defaults to `c_in`. The neck uses the non-default case: the
    bottom-up path concatenates before CAM, and the paper's own trailing 1x1
    ("the Conv1 is the 1x1 convolution layer") is the natural place to bring
    the channel count back down, rather than inserting a block the paper does
    not describe.
    """

    def __init__(self, c_in: int, c_out: int | None = None):
        super().__init__()
        c_out = c_in if c_out is None else c_out
        self.aab = AAB(c_in)
        self.msab = MSAB(c_in)
        self.project = Conv(c_in * 2, c_out, 1)

    def forward(self, x):
        return self.project(torch.cat([self.aab(x), self.msab(x)], 1))


class RM(nn.Module):
    """Refinement Module -- paper Eq. (2), Fig. 3.

        w1 = Sigmoid(Conv(NonLinear(Conv(GAP(F_a)))))
        w2 = Sigmoid(Conv(NonLinear(Conv(GAP(F_b)))))
        w  = w1 + w2
        F_x = Concat[ w * F_a , (1 - w) * F_b ]

    Both inputs must carry the same channel count; the output carries twice
    that. The neck aligns the two sides with 1x1 convolutions beforehand --
    the paper never states the channel bookkeeping.
    """

    # PAPER: w1 and w2 are each a sigmoid, so w = w1 + w2 lies in [0, 2] and
    # (1 - w) can go negative -- i.e. F_b can be *subtracted*. Fig. 3 draws a
    # plain element-wise sum with nothing squashing it afterwards, so "sum" is
    # the faithful default. "mean" ((w1 + w2) / 2) keeps w in [0, 1] and is
    # offered as the first thing to try if training diverges.
    COMBINE_MODES = ("sum", "mean")

    def __init__(self, c: int, reduction: int = 16, combine: str = "sum"):
        super().__init__()
        if combine not in self.COMBINE_MODES:
            raise ValueError(f"combine must be one of {self.COMBINE_MODES}")
        self.combine = combine
        cr = max(c // reduction, 8)
        self.gate_a = self._gate(c, cr)
        self.gate_b = self._gate(c, cr)

    @staticmethod
    def _gate(c: int, cr: int) -> nn.Sequential:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, cr, 1, bias=True),
            nn.SiLU(inplace=True),          # PAPER: "Non-Linear", unspecified
            nn.Conv2d(cr, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, f_a, f_b):
        w = self.gate_a(f_a) + self.gate_b(f_b)
        if self.combine == "mean":
            w = w * 0.5
        return torch.cat([w * f_a, (1.0 - w) * f_b], 1)
