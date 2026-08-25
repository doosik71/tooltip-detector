"""CLAD-Net: CSPDarknet53 backbone + cross-layer aggregation neck + decoupled head.

Two detection classes, both derived from this repository's own annotations
(see common/dataset.py):

    0  tool   the annotated bounding box of the instrument
    1  tip    a square box centred on the annotated tool tip (32 px by default)

The head is anchor-based with a separate objectness branch, decoded the way
YOLOv5 decodes: that is what the paper's loss weights (0.05 / 1.0 / 0.5, the
YOLOv5 defaults), CIoU regression and Obj branch all point to. Fig. 1 draws
the head as three convolutions splitting into Obj / Reg / Cls, i.e. the
decoupled arrangement implemented here.
"""

import math

import torch
import torch.nn as nn

from .backbone import CSPDarknet
from .modules import Conv
from .neck import CrossLayerAggregationNeck

CLASS_NAMES = ("tool", "tip")
STRIDES = (8, 16, 32)

# Anchors in 640-px input space, one row per stride level. Derived from the
# cholec80 train split (k-means over tool boxes plus the constant-size tip
# boxes, then rounded by hand so one P3 anchor sits exactly on the tip box).
# Ratio-matching recall at YOLOv5's threshold of 4.0 is >= 99.5 % and mean
# best IoU >= 0.79 on every split of both datasets.
ANCHORS = (
    ((9, 9), (22, 20), (52, 41)),            # P3 / stride 8
    ((129, 61), (155, 126), (241, 80)),      # P4 / stride 16
    ((240, 189), (309, 128), (405, 223)),    # P5 / stride 32
)


class DecoupledHead(nn.Module):
    """One detection head: shared stem, separate cls and reg/obj branches."""

    def __init__(self, c_in: int, hidden: int, num_anchors: int, num_classes: int):
        super().__init__()
        self.stem = Conv(c_in, hidden, 1)
        self.cls_branch = nn.Sequential(Conv(hidden, hidden, 3), Conv(hidden, hidden, 3))
        self.reg_branch = nn.Sequential(Conv(hidden, hidden, 3), Conv(hidden, hidden, 3))
        self.cls_pred = nn.Conv2d(hidden, num_anchors * num_classes, 1)
        self.reg_pred = nn.Conv2d(hidden, num_anchors * 4, 1)
        self.obj_pred = nn.Conv2d(hidden, num_anchors * 1, 1)

    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_branch(x)
        reg_feat = self.reg_branch(x)
        # Laid out as [x, y, w, h, obj, cls...] so the loss can treat the
        # output exactly like a single-tensor YOLO head.
        return self.reg_pred(reg_feat), self.obj_pred(reg_feat), self.cls_pred(cls_feat)


class CLADNet(nn.Module):
    def __init__(self, num_classes: int = len(CLASS_NAMES), width: float = 0.5,
                 depth: float = 0.33, neck_channels: int = 112, head_hidden: int = 96,
                 rm_combine: str = "sum"):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5
        self.strides = STRIDES
        self.num_anchors = len(ANCHORS[0])

        self.backbone = CSPDarknet(width, depth)
        _, c2, c3, c4, c5 = self.backbone.channels
        self.neck = CrossLayerAggregationNeck((c2, c3, c4, c5), neck_channels, rm_combine)
        self.heads = nn.ModuleList([
            DecoupledHead(c, head_hidden, self.num_anchors, num_classes)
            for c in self.neck.out_channels
        ])

        # anchors are stored in grid units (pixels / stride), as the decode wants them
        anchors = torch.tensor(ANCHORS, dtype=torch.float32)
        anchors = anchors / torch.tensor(STRIDES, dtype=torch.float32).view(-1, 1, 1)
        self.register_buffer("anchors", anchors)          # (nl, na, 2)

        self._initialise_biases()

    def _initialise_biases(self, image_size: int = 640, expected_objects: float = 8.0):
        """YOLOv5's bias prior: start with a low objectness so the huge negative
        majority does not swamp the first epochs."""
        for head, stride in zip(self.heads, self.strides):
            cells = (image_size / stride) ** 2
            obj_bias = math.log(expected_objects / cells)
            nn.init.constant_(head.obj_pred.bias, obj_bias)
            nn.init.constant_(head.cls_pred.bias, math.log(0.6 / (self.num_classes - 0.99)))

    def forward(self, x):
        """Return one raw tensor per level, shaped (B, na, ny, nx, 5 + nc)."""
        _, c2, c3, c4, c5 = self.backbone(x)
        outputs = []
        for feature, head in zip(self.neck(c2, c3, c4, c5), self.heads):
            reg, obj, cls = head(feature)
            b, _, ny, nx = reg.shape
            reg = reg.view(b, self.num_anchors, 4, ny, nx)
            obj = obj.view(b, self.num_anchors, 1, ny, nx)
            cls = cls.view(b, self.num_anchors, self.num_classes, ny, nx)
            out = torch.cat([reg, obj, cls], 2).permute(0, 1, 3, 4, 2).contiguous()
            outputs.append(out)
        return outputs


def decode(outputs, anchors, strides):
    """Turn raw head outputs into absolute-pixel boxes and scores.

    Returns (B, N, 5 + nc) with [cx, cy, w, h, obj, cls...] in input-image
    pixels; obj and cls are already sigmoid-activated.
    """
    decoded = []
    for level, out in enumerate(outputs):
        b, na, ny, nx, no = out.shape
        stride = strides[level]
        device, dtype = out.device, out.dtype

        yv, xv = torch.meshgrid(torch.arange(ny, device=device, dtype=dtype),
                                torch.arange(nx, device=device, dtype=dtype),
                                indexing="ij")
        grid = torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2)
        anchor = anchors[level].to(device=device, dtype=dtype).view(1, na, 1, 1, 2)

        out = out.sigmoid()
        xy = (out[..., 0:2] * 2.0 - 0.5 + grid) * stride
        wh = (out[..., 2:4] * 2.0) ** 2 * anchor * stride
        decoded.append(torch.cat([xy, wh, out[..., 4:]], -1).view(b, -1, no))
    return torch.cat(decoded, 1)


def build(num_classes: int = len(CLASS_NAMES), **kwargs) -> CLADNet:
    return CLADNet(num_classes=num_classes, **kwargs)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
