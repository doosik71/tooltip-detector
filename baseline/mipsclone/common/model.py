"""RTMDet-Pose: one network that finds tool boxes and the tip of each tool.

The ideas come from two OpenMMLab models, rebuilt here in plain PyTorch and
joined into a single end-to-end network:

    image (3, 640, 640)
      │
      ▼ CSPNeXt backbone             RTMDet / RTMPose share it
      C3 (/8) · C4 (/16) · C5 (/32)
      │
      ▼ CSPNeXt-PAFPN neck
      P3 · P4 · P5
      │                    │
      ▼ RTMDet head        ▼ pose branch (RTMPose's head on RoI features)
      tool score           RoIAlign(P3, padded box) → convs → 7×7 conv → K maps
      + box (ltrb)          → flatten → ScaleNorm + Linear → GAU → SimCC x / y
      per prior point        (one 1-D classification per axis and keypoint)

In the original pipeline the two are separate models and the pose model
re-reads a crop of the *image*. Here the pose branch reads a crop of the
detector's own P3 feature map, so the backbone and neck are shared, trained by
both losses, and run once per frame. Training feeds the pose branch jittered
ground-truth boxes; inference feeds it the detector's boxes.

Defaults follow the OpenMMLab configs where they exist (RTMDet-s widths and
depths, SepBN head with shared convs, RTMCCBlock with s = 128 and expansion 2,
KL loss with beta 10). What is this project's own choice is marked as such in
the code: the RoI size, the convs in front of the pose head and the SimCC bin
count have no counterpart in either original.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import batched_nms, roi_align

from .boxes import distance_to_box

STRIDES = (8, 16, 32)

# CSPNeXt depth/width multipliers per RTMDet size, plus whether the box branch
# exponentiates its distances (RTMDet turns it on from `m` up).
SCALES = {
    "tiny": {"deepen": 0.167, "widen": 0.375, "exp_on_reg": False},
    "s": {"deepen": 0.33, "widen": 0.5, "exp_on_reg": False},
    "m": {"deepen": 0.67, "widen": 0.75, "exp_on_reg": True},
    "l": {"deepen": 1.0, "widen": 1.0, "exp_on_reg": True},
}

# ImageNet statistics in RGB order, as RTMDet / RTMPose normalise their input.
_MEAN = (123.675, 116.28, 103.53)
_STD = (58.395, 57.12, 57.375)

DEFAULT_ROI_SIZE = 16        # this project's choice: RoIAlign output, per side
DEFAULT_ROI_PADDING = 1.25   # RTMPose's GetBBoxCenterScale padding
DEFAULT_BINS = 256           # this project's choice: SimCC bins per axis
DEFAULT_NUM_KEYPOINTS = 1    # cholec80 annotates one tip per tool


# ── Building blocks (mmcv ConvModule and friends) ─────────────────────────


class ConvModule(nn.Sequential):
    """Conv (no bias) → BatchNorm → SiLU."""

    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1, groups: int = 1,
                 act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()


class DepthwiseSeparableConv(nn.Sequential):
    """Depthwise k×k ConvModule followed by a pointwise 1×1 ConvModule."""

    def __init__(self, c_in: int, c_out: int, k: int):
        super().__init__()
        self.depthwise_conv = ConvModule(c_in, c_in, k, groups=c_in)
        self.pointwise_conv = ConvModule(c_in, c_out, 1)


class CSPNeXtBlock(nn.Module):
    """3×3 conv, then a 5×5 depthwise-separable conv, with optional identity."""

    def __init__(self, c_in: int, c_out: int, add_identity: bool):
        super().__init__()
        self.conv1 = ConvModule(c_in, c_out, 3)
        self.conv2 = DepthwiseSeparableConv(c_out, c_out, 5)
        self.add_identity = add_identity and c_in == c_out

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class ChannelAttention(nn.Module):
    """Global average pool → 1×1 conv → hard sigmoid gate."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        return x * self.act(self.fc(F.adaptive_avg_pool2d(x, 1)))


class CSPLayer(nn.Module):
    """Cross-stage partial layer built from CSPNeXt blocks."""

    def __init__(self, c_in: int, c_out: int, num_blocks: int, add_identity: bool,
                 channel_attention: bool, expand_ratio: float = 0.5):
        super().__init__()
        mid = int(c_out * expand_ratio)
        self.main_conv = ConvModule(c_in, mid, 1)
        self.short_conv = ConvModule(c_in, mid, 1)
        self.final_conv = ConvModule(2 * mid, c_out, 1)
        self.blocks = nn.Sequential(*[CSPNeXtBlock(mid, mid, add_identity)
                                      for _ in range(num_blocks)])
        self.attention = ChannelAttention(2 * mid) if channel_attention else None

    def forward(self, x):
        out = torch.cat((self.blocks(self.main_conv(x)), self.short_conv(x)), 1)
        if self.attention is not None:
            out = self.attention(out)
        return self.final_conv(out)


class SPPBottleneck(nn.Module):
    """1×1 conv, parallel max-pools (5, 9, 13), concatenate, 1×1 conv."""

    def __init__(self, c_in: int, c_out: int, kernels=(5, 9, 13)):
        super().__init__()
        mid = c_in // 2
        self.conv1 = ConvModule(c_in, mid, 1)
        self.poolings = nn.ModuleList(nn.MaxPool2d(k, 1, k // 2) for k in kernels)
        self.conv2 = ConvModule(mid * (len(kernels) + 1), c_out, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.poolings], 1))


# ── Backbone, neck, detection head ────────────────────────────────────────


class CSPNeXt(nn.Module):
    """CSPNeXt P5: a 3-conv stem and four stages, returning stages 2-4."""

    # in, out, blocks, add_identity, use_spp
    ARCH = ((64, 128, 3, True, False), (128, 256, 6, True, False),
            (256, 512, 6, True, False), (512, 1024, 3, False, True))

    def __init__(self, deepen: float, widen: float):
        super().__init__()
        stem = int(self.ARCH[0][0] * widen)
        self.stem = nn.Sequential(ConvModule(3, stem // 2, 3, 2),
                                  ConvModule(stem // 2, stem // 2, 3),
                                  ConvModule(stem // 2, stem, 3))
        self.stages = nn.ModuleList()
        for c_in, c_out, blocks, identity, spp in self.ARCH:
            c_in, c_out = int(c_in * widen), int(c_out * widen)
            layers = [ConvModule(c_in, c_out, 3, 2)]
            if spp:
                layers.append(SPPBottleneck(c_out, c_out))
            layers.append(CSPLayer(c_out, c_out, max(round(blocks * deepen), 1),
                                   add_identity=identity, channel_attention=True))
            self.stages.append(nn.Sequential(*layers))
        self.out_channels = [int(c * widen) for c in (256, 512, 1024)]

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i >= 1:
                outs.append(x)
        return outs


class CSPNeXtPAFPN(nn.Module):
    """Top-down then bottom-up fusion of three levels, 3×3 conv per output."""

    def __init__(self, in_channels: list[int], out_channels: int, num_blocks: int):
        super().__init__()
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for i in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(ConvModule(in_channels[i], in_channels[i - 1], 1))
            self.top_down_blocks.append(CSPLayer(in_channels[i - 1] * 2, in_channels[i - 1],
                                                 num_blocks, False, False))
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.downsamples.append(ConvModule(in_channels[i], in_channels[i], 3, 2))
            self.bottom_up_blocks.append(CSPLayer(in_channels[i] * 2, in_channels[i + 1],
                                                  num_blocks, False, False))
        self.out_convs = nn.ModuleList(ConvModule(c, out_channels, 3) for c in in_channels)

    def forward(self, inputs):
        n = len(inputs)
        inner = [inputs[-1]]
        for i in range(n - 1, 0, -1):
            high = self.reduce_layers[n - 1 - i](inner[0])
            inner[0] = high
            up = F.interpolate(high, scale_factor=2, mode="nearest")
            inner.insert(0, self.top_down_blocks[n - 1 - i](torch.cat([up, inputs[i - 1]], 1)))
        outs = [inner[0]]
        for i in range(n - 1):
            down = self.downsamples[i](outs[-1])
            outs.append(self.bottom_up_blocks[i](torch.cat([down, inner[i + 1]], 1)))
        return [conv(out) for conv, out in zip(self.out_convs, outs)]


class RTMDetSepBNHead(nn.Module):
    """Classification and box branches whose convs are shared across levels
    while each level keeps its own BatchNorm and prediction layer."""

    def __init__(self, num_classes: int, channels: int, stacked_convs: int = 2,
                 exp_on_reg: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.exp_on_reg = exp_on_reg
        levels = len(STRIDES)
        self.cls_convs = nn.ModuleList(nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
                                       for _ in range(stacked_convs))
        self.reg_convs = nn.ModuleList(nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
                                       for _ in range(stacked_convs))
        self.cls_bns = nn.ModuleList(nn.ModuleList(nn.BatchNorm2d(channels)
                                                   for _ in range(stacked_convs))
                                     for _ in range(levels))
        self.reg_bns = nn.ModuleList(nn.ModuleList(nn.BatchNorm2d(channels)
                                                   for _ in range(stacked_convs))
                                     for _ in range(levels))
        self.rtm_cls = nn.ModuleList(nn.Conv2d(channels, num_classes, 1) for _ in range(levels))
        self.rtm_reg = nn.ModuleList(nn.Conv2d(channels, 4, 1) for _ in range(levels))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        prior_bias = -math.log((1 - 0.01) / 0.01)
        for conv in self.rtm_cls:
            nn.init.constant_(conv.bias, prior_bias)

    def forward(self, feats):
        cls_scores, box_dists = [], []
        for level, (x, stride) in enumerate(zip(feats, STRIDES)):
            c, r = x, x
            for conv, bn in zip(self.cls_convs, self.cls_bns[level]):
                c = F.silu(bn(conv(c)))
            for conv, bn in zip(self.reg_convs, self.reg_bns[level]):
                r = F.silu(bn(conv(r)))
            dist = self.rtm_reg[level](r)
            dist = dist.exp() if self.exp_on_reg else dist
            cls_scores.append(self.rtm_cls[level](c))
            box_dists.append(dist * stride)
        return cls_scores, box_dists


# ── Pose branch (RTMPose's RTMCC head) ────────────────────────────────────


class ScaleNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.linalg.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class GatedAttentionUnit(nn.Module):
    """RTMPose's RTMCCBlock (self-attention, no relative position bias).

    Attention runs across the K keypoint tokens. With K = 1 the attention map
    is a single non-negative scalar per instance, so the block reduces to a
    gated MLP with a residual connection -- it still trains, it just has no
    keypoint relations to model.
    """

    def __init__(self, dim: int, expansion: int = 2, s: int = 128):
        super().__init__()
        self.e, self.s = dim * expansion, s
        self.ln = ScaleNorm(dim)
        self.uv = nn.Linear(dim, 2 * self.e + s, bias=False)
        self.gamma = nn.Parameter(torch.rand(2, s))
        self.beta = nn.Parameter(torch.rand(2, s))
        self.o = nn.Linear(self.e, dim, bias=False)
        self.res_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        h = F.silu(self.uv(self.ln(x)))
        u, v, base = torch.split(h, [self.e, self.e, self.s], dim=-1)
        base = base.unsqueeze(2) * self.gamma + self.beta
        q, k = base.unbind(2)
        kernel = F.relu(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.s)).square()
        return x * self.res_scale + self.o(u * torch.bmm(kernel, v))


class PoseHead(nn.Module):
    """RoI features → K keypoints as SimCC x / y classifications."""

    def __init__(self, channels: int, num_keypoints: int, roi_size: int, bins: int,
                 num_convs: int = 4, hidden: int = 256):
        super().__init__()
        # This project's choice: RTMPose reads a 768-channel stride-32 map of
        # a dedicated backbone; here the input is a shared, much thinner
        # neck map, so the branch gets a few convs of its own first.
        self.convs = nn.Sequential(*[ConvModule(channels, channels, 3) for _ in range(num_convs)])
        self.final_layer = nn.Conv2d(channels, num_keypoints, 7, 1, 3)
        flat = roi_size * roi_size
        self.mlp = nn.Sequential(ScaleNorm(flat), nn.Linear(flat, hidden, bias=False))
        self.gau = GatedAttentionUnit(hidden)
        self.cls_x = nn.Linear(hidden, bins, bias=False)
        self.cls_y = nn.Linear(hidden, bins, bias=False)
        # RTMCCHead's initialisation
        nn.init.normal_(self.final_layer.weight, 0, 0.001)
        nn.init.zeros_(self.final_layer.bias)
        for m in (self.mlp[1], self.gau.uv, self.gau.o, self.cls_x, self.cls_y):
            nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, roi_feats):
        x = self.final_layer(self.convs(roi_feats)).flatten(2)     # (R, K, r*r)
        x = self.gau(self.mlp(x))
        return self.cls_x(x), self.cls_y(x)                        # (R, K, bins) each


# ── The joint model ───────────────────────────────────────────────────────


class RTMDetPose(nn.Module):
    def __init__(self, scale: str = "s", num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
                 roi_size: int = DEFAULT_ROI_SIZE, roi_padding: float = DEFAULT_ROI_PADDING,
                 bins: int = DEFAULT_BINS):
        super().__init__()
        if scale not in SCALES:
            raise ValueError(f"scale must be one of {tuple(SCALES)}")
        cfg = SCALES[scale]
        self.scale = scale
        self.num_keypoints = num_keypoints
        self.roi_size = roi_size
        self.roi_padding = roi_padding
        self.bins = bins

        self.backbone = CSPNeXt(cfg["deepen"], cfg["widen"])
        channels = int(256 * cfg["widen"])
        self.neck = CSPNeXtPAFPN(self.backbone.out_channels, channels,
                                 max(round(3 * cfg["deepen"]), 1))
        self.head = RTMDetSepBNHead(1, channels, exp_on_reg=cfg["exp_on_reg"])
        self.pose_head = PoseHead(channels, num_keypoints, roi_size, bins)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False)
        self._prior_cache: dict = {}

    def arch(self) -> dict:
        """Everything `build` needs to recreate this model."""
        return {"scale": self.scale, "num_keypoints": self.num_keypoints,
                "roi_size": self.roi_size, "roi_padding": self.roi_padding, "bins": self.bins}

    def forward(self, images: torch.Tensor):
        """images: (B, 3, H, W) in 0..255. Returns (neck features, cls, box dists)."""
        x = (images - self.mean) / self.std
        feats = self.neck(self.backbone(x))
        cls_scores, box_dists = self.head(feats)
        return feats, cls_scores, box_dists

    def priors(self, feats) -> torch.Tensor:
        """(N, 3) [x, y, stride] over all levels, offset 0 (RTMDet's point generator)."""
        key = tuple((f.shape[-2], f.shape[-1]) for f in feats) + (feats[0].device,)
        if key not in self._prior_cache:
            rows = []
            for f, stride in zip(feats, STRIDES):
                h, w = f.shape[-2:]
                ys, xs = torch.meshgrid(torch.arange(h, device=f.device),
                                        torch.arange(w, device=f.device), indexing="ij")
                rows.append(torch.stack([xs.flatten() * stride, ys.flatten() * stride,
                                         torch.full((h * w,), stride, device=f.device)], 1))
            self._prior_cache[key] = torch.cat(rows).float()
        return self._prior_cache[key]

    @staticmethod
    def flatten(cls_scores, box_dists):
        """Per-level maps → (B, N, C) logits and (B, N, 4) distances."""
        b = cls_scores[0].shape[0]
        cls = torch.cat([c.permute(0, 2, 3, 1).reshape(b, -1, c.shape[1]) for c in cls_scores], 1)
        dist = torch.cat([d.permute(0, 2, 3, 1).reshape(b, -1, 4) for d in box_dists], 1)
        return cls, dist

    def rois(self, boxes: torch.Tensor) -> torch.Tensor:
        """Pad xyxy boxes about their centre by `roi_padding`."""
        centre = (boxes[:, :2] + boxes[:, 2:]) / 2
        half = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1.0) * (self.roi_padding / 2)
        return torch.cat([centre - half, centre + half], 1)

    def pose(self, p3: torch.Tensor, batch_index: torch.Tensor, rois: torch.Tensor):
        """SimCC logits for RoIs (already padded) on the stride-8 map.

        Runs in fp32 even when the trunk ran under autocast: the branch is
        small, and the loss is computed outside autocast anyway."""
        feats = roi_align(p3.float(), torch.cat([batch_index[:, None].to(rois), rois], 1).float(),
                          output_size=self.roi_size, spatial_scale=1.0 / STRIDES[0],
                          sampling_ratio=2, aligned=True)
        return self.pose_head(feats)

    def decode_keypoints(self, simcc_x, simcc_y, rois, beta: float = 10.0):
        """(R, K, bins) logits → (R, K, 2) input-pixel coordinates and (R, K) scores.

        Bin i covers [i, i + 1) / bins of the RoI, so its centre is the
        estimate. The score is the smaller of the two axes' peak probability
        under the same softmax temperature the loss uses.
        """
        px = F.softmax(simcc_x.float() * beta, -1)
        py = F.softmax(simcc_y.float() * beta, -1)
        vx, ix = px.max(-1)
        vy, iy = py.max(-1)
        size = (rois[:, 2:] - rois[:, :2])[:, None, :]
        rel = torch.stack([(ix.float() + 0.5) / self.bins, (iy.float() + 0.5) / self.bins], -1)
        return rois[:, None, :2] + rel * size, torch.minimum(vx, vy)

    @torch.no_grad()
    def predict(self, images: torch.Tensor, conf: float = 0.25, max_det: int = 100,
                nms_iou: float = 0.65, pre_nms: int = 1000, pose_conf: float | None = None):
        """Detect tools and their tips.

        Returns one dict per image with, in input pixels,
            boxes      (n, 4) xyxy
            scores     (n,)
            keypoints  (n, K, 2)   only for boxes scoring >= pose_conf
            kpt_scores (n, K)      NaN where no keypoint was computed
        `pose_conf` defaults to `conf`; set it higher than `conf` to run the
        pose branch only on the boxes a tip metric will actually read.
        """
        feats, cls_scores, box_dists = self(images)
        cls, dist = self.flatten(cls_scores, box_dists)
        priors = self.priors(feats)
        scores = cls.float().sigmoid()[..., 0]
        boxes = distance_to_box(priors[None, :, :2], dist.float())
        pose_conf = conf if pose_conf is None else pose_conf

        results, keep_boxes, keep_index = [], [], []
        for i in range(images.shape[0]):
            s, b = scores[i], boxes[i]
            mask = s >= conf
            s, b = s[mask], b[mask]
            if s.numel() > pre_nms:
                s, top = s.topk(pre_nms)
                b = b[top]
            keep = batched_nms(b, s, torch.zeros_like(s, dtype=torch.long), nms_iou)[:max_det]
            b, s = b[keep], s[keep]
            results.append({"boxes": b, "scores": s})
            posed = s >= pose_conf
            keep_boxes.append(b[posed])
            keep_index.append(torch.full((int(posed.sum()),), i, device=b.device))

        for r in results:
            r["keypoints"] = r["boxes"].new_full((len(r["boxes"]), self.num_keypoints, 2), float("nan"))
            r["kpt_scores"] = r["boxes"].new_full((len(r["boxes"]), self.num_keypoints), float("nan"))
        all_boxes = torch.cat(keep_boxes)
        if len(all_boxes):
            rois = self.rois(all_boxes)
            sx, sy = self.pose(feats[0], torch.cat(keep_index), rois)
            kpts, kscores = self.decode_keypoints(sx, sy, rois)
            start = 0
            for r, b in zip(results, keep_boxes):
                n = len(b)
                r["keypoints"][:n] = kpts[start:start + n]       # posed boxes lead (sorted by score)
                r["kpt_scores"][:n] = kscores[start:start + n]
                start += n
        return results


def build(**arch) -> RTMDetPose:
    return RTMDetPose(**arch)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
