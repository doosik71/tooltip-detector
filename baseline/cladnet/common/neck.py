"""The Cross-Layer Aggregated Attention Module -- CLAD-Net's neck (paper Fig. 1).

Three paths, in the order the figure draws them:

  top-down    P5 <- C5,  P4 <- concat[up(P5), C4],  P3 <- ...,  P2 <- ...
  lateral     C3/C4/C5 -> depthwise separable conv -> RM(., P3/P4/P5)
  bottom-up   N2 <- P2,  N{k+1} <- CAM(concat[down(N_k), RM_k])

`PAPER:` comments mark every place the text is silent or contradicts Fig. 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import C3
from .modules import CAM, RM, Conv, DWConv


class CrossLayerAggregationNeck(nn.Module):
    """Takes C2..C5 from the backbone, returns N3, N4, N5 for the head."""

    def __init__(self, in_channels: tuple[int, int, int, int], channels: int = 96,
                 rm_combine: str = "sum"):
        super().__init__()
        c2, c3, c4, c5 = in_channels
        c = channels

        # PAPER: the paper never states the neck's channel bookkeeping. Every
        # branch is brought to a single width `c` with 1x1 convolutions so RM
        # (which requires both inputs to match) and the concatenations line up.
        self.align5 = Conv(c5, c, 1)
        self.align4 = Conv(c4, c, 1)
        self.align3 = Conv(c3, c, 1)
        self.align2 = Conv(c2, c, 1)

        self.td4 = C3(2 * c, c, n=1, shortcut=False)
        self.td3 = C3(2 * c, c, n=1, shortcut=False)
        self.td2 = C3(2 * c, c, n=1, shortcut=False)

        # Lateral cross-layer connections (the orange arrows in Fig. 1):
        # "we first pass the feature maps C3, C4, C5 extracted from the
        # backbone via depth-separable convolutional transfer".
        self.lat3 = DWConv(c3, c)
        self.lat4 = DWConv(c4, c)
        self.lat5 = DWConv(c5, c)

        self.rm3 = RM(c, combine=rm_combine)
        self.rm4 = RM(c, combine=rm_combine)
        self.rm5 = RM(c, combine=rm_combine)

        # PAPER: the text says the bottom-up path *upsamples* N2/N3/N4 to the
        # size of C3/C4/C5, but that direction lowers the resolution and Fig.
        # 1's legend marks those arrows "Downsample". The figure is followed.
        self.down3 = Conv(c, c, 3, 2)
        self.down4 = Conv(c, c, 3, 2)
        self.down5 = Conv(c, c, 3, 2)

        # concat[down(N), RM] is c + 2c wide; CAM's own trailing 1x1 brings it
        # back to c (see common.modules.CAM).
        self.cam3 = CAM(3 * c, c)
        self.cam4 = CAM(3 * c, c)
        self.cam5 = CAM(3 * c, c)

        self.out_channels = (c, c, c)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def forward(self, c2, c3, c4, c5):
        # top-down
        p5 = self.align5(c5)
        p4 = self.td4(torch.cat([self._up(p5, c4), self.align4(c4)], 1))
        p3 = self.td3(torch.cat([self._up(p4, c3), self.align3(c3)], 1))
        p2 = self.td2(torch.cat([self._up(p3, c2), self.align2(c2)], 1))

        # lateral refinement
        r3 = self.rm3(self.lat3(c3), p3)
        r4 = self.rm4(self.lat4(c4), p4)
        r5 = self.rm5(self.lat5(c5), p5)

        # bottom-up. PAPER: Fig. 1 feeds only N3/N4/N5 to the head, so N2 == P2
        # exists purely to seed the bottom-up path.
        n2 = p2
        n3 = self.cam3(torch.cat([self.down3(n2), r3], 1))
        n4 = self.cam4(torch.cat([self.down4(n3), r4], 1))
        n5 = self.cam5(torch.cat([self.down5(n4), r5], 1))
        return n3, n4, n5
