"""CSPDarknet53 backbone.

The paper states only "CLAD-Net uses the lightweight CSPdarkernet53 as the
backbone feature extraction network" and reports 7.5 M parameters for the whole
network -- YOLOv5s territory. This is the YOLOv5 CSPDarknet (C3 bottleneck
blocks + SPPF), with depth and width multipliers so the total can be tuned to
land on the published parameter count.

Returns the five stages C1..C5 at strides 2, 4, 8, 16, 32. The neck consumes
C2..C5; C1 exists only because Fig. 1 draws it.
"""

import torch
import torch.nn as nn

from .modules import Conv


class Bottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c_, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP bottleneck with 3 convolutions (YOLOv5's C3)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c1, c_, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, 1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], 1))


class SPPF(nn.Module):
    """Spatial pyramid pooling, fast variant."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c_ * 4, c2, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


def _round_channels(c: int, width: float, divisor: int = 8) -> int:
    c = int(c * width)
    return max(divisor, (c + divisor // 2) // divisor * divisor)


class CSPDarknet(nn.Module):
    """CSPDarknet53 feature extractor returning C1..C5."""

    def __init__(self, width: float = 0.5, depth: float = 0.33):
        super().__init__()
        c1, c2, c3, c4, c5 = (_round_channels(c, width) for c in (64, 128, 256, 512, 1024))
        n2, n3, n4, n5 = (max(round(n * depth), 1) for n in (3, 6, 9, 3))

        self.stem = Conv(3, c1, 6, 2, 2)                       # C1, stride 2
        self.stage2 = nn.Sequential(Conv(c1, c2, 3, 2), C3(c2, c2, n2))    # C2, stride 4
        self.stage3 = nn.Sequential(Conv(c2, c3, 3, 2), C3(c3, c3, n3))    # C3, stride 8
        self.stage4 = nn.Sequential(Conv(c3, c4, 3, 2), C3(c4, c4, n4))    # C4, stride 16
        self.stage5 = nn.Sequential(Conv(c4, c5, 3, 2), C3(c5, c5, n5),
                                    SPPF(c5, c5))                          # C5, stride 32
        self.channels = (c1, c2, c3, c4, c5)

    def forward(self, x):
        p1 = self.stem(x)
        p2 = self.stage2(p1)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p1, p2, p3, p4, p5
