"""detrtooltip: a ResNet + FPN backbone with a Deformable DETR head that gives
each surgical tool a box and one or two tips.

    image 736 x 480 (normalised RGB)
      -> ResNet encoder (C2..C5)  -> FPN (P2..P5, 256 ch)          backbone
      -> Deformable DETR encoder on P3..P5                          head
      -> two-stage proposals (top `num_queries` tokens)
      -> Deformable DETR decoder, iterative box and tip refinement
      -> per query: tool score, box, tip slots A and B, P(B exists)
      -> tip refinement: soft-argmax over a P2 window around each tip

The backbone is what stage 1 (masked reconstruction, `Pretrainer`) trains and
what stage 2a and the interactive trainer keep frozen; everything else is the
head. Tip slots carry no order: the loss matches them to the label's tips
either way round (see loss.py). Slot A always exists; slot B is reported as
`tip2` only when P(B exists) > 0.5.
"""

import math

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from .boxes import inverse_sigmoid
from .msdeform import MSDeformAttn

CLASS_NAMES = ("tool",)
IMAGE_WIDTH, IMAGE_HEIGHT = 736, 480
# Channel statistics of 2,000 random cholec80 frames (all splits), in 0-1 RGB.
PIXEL_MEAN = (0.2935, 0.1911, 0.1876)
PIXEL_STD = (0.2601, 0.2101, 0.2071)

ENCODER_CHANNELS = {50: (256, 512, 1024, 2048), 101: (256, 512, 1024, 2048)}
DEFAULT_ARCH = {
    "depth": 50,            # ResNet depth: 50 or 101
    "d_model": 256,
    "heads": 8,
    "points": 4,            # deformable sampling points per head and level
    "ffn": 1024,
    "enc_layers": 6,
    "dec_layers": 6,
    "num_queries": 30,
    "refine_window": 15,    # tip refinement window side, in P2 cells (4 px each)
}


# ---------------------------------------------------------------------------
# Backbone: ResNet encoder + FPN decoder
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    """ResNet encoder (random init) and an FPN decoder; returns P2..P5."""

    def __init__(self, depth: int = 50, channels: int = 256):
        super().__init__()
        if depth not in ENCODER_CHANNELS:
            raise ValueError(f"ResNet depth must be one of {tuple(ENCODER_CHANNELS)}")
        resnet = getattr(torchvision.models, f"resnet{depth}")(weights=None)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layers = nn.ModuleList([resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4])
        self.lateral = nn.ModuleList(nn.Conv2d(c, channels, 1) for c in ENCODER_CHANNELS[depth])
        self.output = nn.ModuleList(nn.Conv2d(channels, channels, 3, padding=1)
                                    for _ in ENCODER_CHANNELS[depth])
        self.register_buffer("mean", torch.tensor(PIXEL_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(PIXEL_STD).view(1, 3, 1, 1), persistent=False)

    def normalise(self, images: torch.Tensor) -> torch.Tensor:
        """0-1 RGB -> zero-mean, unit-variance input."""
        return (images - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Already normalised input -> [P2, P3, P4, P5]."""
        features = []
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        laterals = [conv(f) for conv, f in zip(self.lateral, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
        return [conv(f) for conv, f in zip(self.output, laterals)]


class ReconstructionHead(nn.Module):
    """Stage 1 only: P2 (stride 4) -> full-resolution pixels."""

    def __init__(self, channels: int = 256, scale: int = 4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, 128, 3, padding=1), nn.GroupNorm(32, 128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 3 * scale * scale, 1))
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, p2: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.body(p2))


class Pretrainer(nn.Module):
    """Stage 1, SimMIM-style: masked patches of the normalised input are
    replaced by a learned per-channel value, and the backbone plus a light
    head predicts the pixels (see loss.ReconstructionLoss)."""

    def __init__(self, depth: int = 50, **_unused):
        super().__init__()
        self.backbone = Backbone(depth)
        self.head = ReconstructionHead()
        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """images (B, 3, H, W) 0-1 RGB, mask (B, 1, H, W) 1 where hidden ->
        prediction (B, 3, H, W) of the per-patch normalised pixels."""
        x = self.backbone.normalise(images)
        x = x * (1 - mask) + self.mask_token * mask
        return self.head(self.backbone(x)[0])


# ---------------------------------------------------------------------------
# Transformer pieces
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


def sine_position_map(h: int, w: int, d_model: int, device) -> torch.Tensor:
    """DETR's normalised sine embedding of a feature map's cells -> (h*w, d_model)."""
    feats = d_model // 2
    y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2 * math.pi
    x = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2 * math.pi
    dim_t = 10000 ** (2 * (torch.arange(feats, device=device) // 2) / feats)
    pos_x = x[:, None] / dim_t
    pos_y = y[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), 2).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), 2).flatten(1)
    return torch.cat((pos_y[:, None, :].expand(h, w, feats),
                      pos_x[None, :, :].expand(h, w, feats)), -1).flatten(0, 1)


def sine_box_embedding(boxes: torch.Tensor, d_model: int) -> torch.Tensor:
    """(..., 4) normalised cx, cy, w, h -> (..., 2 * d_model), as in DAB/DINO."""
    feats = d_model // 2
    dim_t = 10000 ** (2 * (torch.arange(feats, device=boxes.device) // 2) / feats)
    parts = []
    for i in range(4):
        pos = boxes[..., i, None] * 2 * math.pi / dim_t
        parts.append(torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), -1).flatten(-2))
    return torch.cat(parts, -1)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, points: int, ffn: int, levels: int):
        super().__init__()
        self.attn = MSDeformAttn(d_model, levels, heads, points)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn), nn.ReLU(inplace=True),
                                 nn.Linear(ffn, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, shapes):
        src = self.norm1(src + self.attn(src + pos, reference_points, src, shapes))
        return self.norm2(src + self.ffn(src))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, points: int, ffn: int, levels: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformAttn(d_model, levels, heads, points)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn), nn.ReLU(inplace=True),
                                 nn.Linear(ffn, d_model))
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, query_pos, reference_boxes, memory, shapes):
        q = k = tgt + query_pos
        tgt = self.norm1(tgt + self.self_attn(q, k, tgt, need_weights=False)[0])
        levels = len(shapes)
        refs = reference_boxes[:, :, None, :].expand(-1, -1, levels, -1)
        tgt = self.norm2(tgt + self.cross_attn(tgt + query_pos, refs, memory, shapes))
        return self.norm3(tgt + self.ffn(tgt))


class TipRefiner(nn.Module):
    """Look again at each tip on P2 (stride 4): sample a `window` x `window`
    grid around it, turn it into a heatmap and move the tip to the heatmap's
    soft-argmax. Regression alone is coarse at a few pixels; this is the step
    meant to recover them."""

    def __init__(self, d_model: int, window: int, stride: int = 4):
        super().__init__()
        self.window = window
        self.stride = stride
        self.reduce = nn.Conv2d(d_model, 64, 1)
        self.query_proj = nn.Linear(d_model, 64)
        self.body = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1))
        offsets = (torch.arange(window, dtype=torch.float32) - (window - 1) / 2) * stride
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        # (window*window, 2) pixel offsets from the tip, x then y
        self.register_buffer("cell_offsets", torch.stack((ox, oy), -1).flatten(0, 1),
                             persistent=False)

    def forward(self, p2: torch.Tensor, queries: torch.Tensor, tips: torch.Tensor) -> torch.Tensor:
        """p2 (B, C, h, w); queries (B, Q, C); tips (B, Q, 2, 2) normalised ->
        refined tips (B, Q, 2, 2). The input tips are treated as constants:
        only the offset this module adds is learned here."""
        batch, queries_n = tips.shape[:2]
        size = p2.new_tensor([IMAGE_WIDTH, IMAGE_HEIGHT])
        centres = tips.detach().reshape(batch, queries_n * 2, 1, 2)
        points = centres + self.cell_offsets.to(p2.dtype) / size                  # (B, Q*2, K*K, 2)
        features = self.reduce(p2)
        with torch.autocast(device_type=p2.device.type, enabled=False):
            sampled = F.grid_sample(features.float(), 2 * points.float() - 1, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)  # (B, 64, Q*2, K*K)
        sampled = sampled.to(features.dtype).permute(0, 2, 1, 3)                  # (B, Q*2, 64, K*K)
        context = self.query_proj(queries).repeat_interleave(2, dim=1)            # (B, Q*2, 64)
        sampled = sampled + context[..., None]
        k = self.window
        logits = self.body(sampled.reshape(batch * queries_n * 2, 64, k, k)).flatten(1)
        weights = logits.float().softmax(-1)                                        # (N, K*K)
        offset = weights @ self.cell_offsets                                         # (N, 2) px
        offset = offset.view(batch, queries_n, 2, 2) / size.float()
        return tips.detach() + offset.to(tips.dtype)


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------

class DetrTooltip(nn.Module):
    LEVELS = 3      # P3, P4, P5

    def __init__(self, depth: int = 50, d_model: int = 256, heads: int = 8, points: int = 4,
                 ffn: int = 1024, enc_layers: int = 6, dec_layers: int = 6,
                 num_queries: int = 30, refine_window: int = 15):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.backbone = Backbone(depth, d_model)

        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(d_model, d_model, 1), nn.GroupNorm(32, d_model))
            for _ in range(self.LEVELS))
        self.level_embed = nn.Parameter(torch.zeros(self.LEVELS, d_model))
        nn.init.normal_(self.level_embed)
        self.encoder = nn.ModuleList(EncoderLayer(d_model, heads, points, ffn, self.LEVELS)
                                     for _ in range(enc_layers))

        # Two-stage proposals: every encoder token scores itself as a tool
        # and refines an anchor box around its cell.
        self.enc_output = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.enc_class = nn.Linear(d_model, 1)
        self.enc_box = MLP(d_model, d_model, 4, 3)

        self.query_content = nn.Embedding(num_queries, d_model)
        self.query_pos_head = MLP(2 * d_model, d_model, d_model, 2)
        self.decoder = nn.ModuleList(DecoderLayer(d_model, heads, points, ffn, self.LEVELS)
                                     for _ in range(dec_layers))
        self.class_heads = nn.ModuleList(nn.Linear(d_model, 1) for _ in range(dec_layers))
        self.box_heads = nn.ModuleList(MLP(d_model, d_model, 4, 3) for _ in range(dec_layers))
        self.tip_heads = nn.ModuleList(MLP(d_model, d_model, 4, 3) for _ in range(dec_layers))
        self.exist_heads = nn.ModuleList(nn.Linear(d_model, 1) for _ in range(dec_layers))
        self.refiner = TipRefiner(d_model, refine_window)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        prior = -math.log((1 - 0.01) / 0.01)        # start with P(tool) = 0.01
        for head in list(self.class_heads) + [self.enc_class]:
            nn.init.constant_(head.bias, prior)
        for mlp in list(self.box_heads) + list(self.tip_heads) + [self.enc_box]:
            nn.init.constant_(mlp.layers[-1].weight, 0.0)
            nn.init.constant_(mlp.layers[-1].bias, 0.0)
        for head in self.exist_heads:
            nn.init.constant_(head.bias, prior)     # a second tip is rare

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _anchors(shapes, device) -> torch.Tensor:
        """Inverse-sigmoid anchor boxes, one per encoder token."""
        anchors = []
        for level, (h, w) in enumerate(shapes):
            ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                                    torch.arange(w, device=device, dtype=torch.float32),
                                    indexing="ij")
            centre = torch.stack(((xs + 0.5) / w, (ys + 0.5) / h), -1).flatten(0, 1)
            size = torch.full_like(centre, 0.05 * 2.0 ** level)
            anchors.append(torch.cat((centre, size), -1))
        return inverse_sigmoid(torch.cat(anchors, 0))

    @staticmethod
    def _reference_points(shapes, device) -> torch.Tensor:
        """Normalised cell centres of every encoder token, (S, 2)."""
        points = []
        for h, w in shapes:
            ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                                    torch.arange(w, device=device, dtype=torch.float32),
                                    indexing="ij")
            points.append(torch.stack(((xs + 0.5) / w, (ys + 0.5) / h), -1).flatten(0, 1))
        return torch.cat(points, 0)

    def head_modules(self) -> list[nn.Module]:
        """Everything but the backbone."""
        return [module for name, module in self.named_children() if name != "backbone"]

    # -- forward --------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> dict:
        """images (B, 3, 480, 736) 0-1 RGB -> outputs in normalised coordinates:

            logits (B, Q), boxes (B, Q, 4) cx cy w h, tips (B, Q, 2, 2) refined,
            tips_coarse (B, Q, 2, 2), exist_logits (B, Q),
            aux: the same keys (no refinement) for every earlier decoder layer,
            enc: logits and boxes of the selected two-stage proposals.
        """
        batch = images.shape[0]
        p2, *pyramid = self.backbone(self.backbone.normalise(images))

        shapes, sources, positions = [], [], []
        for level, (proj, feature) in enumerate(zip(self.input_proj, pyramid)):
            h, w = feature.shape[-2:]
            shapes.append((h, w))
            sources.append(proj(feature).flatten(2).transpose(1, 2))
            pos = sine_position_map(h, w, self.d_model, images.device) + self.level_embed[level]
            positions.append(pos.to(sources[-1].dtype)[None].expand(batch, -1, -1))
        memory = torch.cat(sources, 1)
        pos = torch.cat(positions, 1)

        enc_refs = self._reference_points(shapes, images.device)
        enc_refs = enc_refs[None, :, None, :].expand(batch, -1, self.LEVELS, -1)
        for layer in self.encoder:
            memory = layer(memory, pos, enc_refs, shapes)

        # Two-stage proposals -> initial reference boxes of the queries.
        enc_memory = self.enc_output(memory)
        enc_logits = self.enc_class(enc_memory)[..., 0]
        enc_boxes_unact = self.enc_box(enc_memory) + self._anchors(shapes, images.device)
        top = enc_logits.topk(self.num_queries, dim=1).indices
        top_boxes_unact = enc_boxes_unact.gather(1, top[..., None].expand(-1, -1, 4))
        enc = {"logits": enc_logits.gather(1, top), "boxes": top_boxes_unact.sigmoid()}

        reference_boxes = top_boxes_unact.sigmoid().detach()
        reference_tips = reference_boxes[..., :2].repeat(1, 1, 2)     # both tips at the centre
        tgt = self.query_content.weight[None].expand(batch, -1, -1)

        layers_out = []
        for i, layer in enumerate(self.decoder):
            query_pos = self.query_pos_head(sine_box_embedding(reference_boxes, self.d_model)
                                            .to(tgt.dtype))
            tgt = layer(tgt, query_pos, reference_boxes, memory, shapes)
            boxes = (inverse_sigmoid(reference_boxes) + self.box_heads[i](tgt)).sigmoid()
            tips = (inverse_sigmoid(reference_tips) + self.tip_heads[i](tgt)).sigmoid()
            layers_out.append({
                "logits": self.class_heads[i](tgt)[..., 0],
                "boxes": boxes,
                "tips": tips.view(batch, -1, 2, 2),
                "exist_logits": self.exist_heads[i](tgt)[..., 0],
            })
            reference_boxes, reference_tips = boxes.detach(), tips.detach()

        final = layers_out[-1]
        refined = self.refiner(p2, tgt, final["tips"])
        return {
            "logits": final["logits"], "boxes": final["boxes"], "tips": refined,
            "tips_coarse": final["tips"], "exist_logits": final["exist_logits"],
            "aux": layers_out[:-1], "enc": enc,
        }


def build(**arch) -> DetrTooltip:
    return DetrTooltip(**{**DEFAULT_ARCH, **arch})


def parameter_count(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


TIP_MERGE_PX = 20.0


@torch.no_grad()
def postprocess(outputs: dict, conf: float, width: int = IMAGE_WIDTH,
                height: int = IMAGE_HEIGHT) -> list[list[dict]]:
    """Model outputs -> per image, annotations in frame pixels, best first:
    {"bbox", "tip", optional "tip2", "score", "tip2_score"}. One query is one
    tool, so no NMS is applied. Two tips at most TIP_MERGE_PX apart are given
    as one `tip` at their mean, as the labels record them (the 20 px
    tolerance of tip F1 @ 20 px)."""
    scores = outputs["logits"].float().sigmoid()
    exist = outputs["exist_logits"].float().sigmoid()
    boxes = outputs["boxes"].float()
    tips = outputs["tips"].float()
    scale = boxes.new_tensor([width, height])
    results = []
    for b in range(scores.shape[0]):
        keep = (scores[b] >= conf).nonzero().flatten()
        keep = keep[scores[b, keep].argsort(descending=True)]
        annotations = []
        for q in keep.tolist():
            cx, cy, w, h = (boxes[b, q] * scale.repeat(2)).tolist()
            x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
            x2, y2 = min(float(width), cx + w / 2), min(float(height), cy + h / 2)
            tip_points = (tips[b, q] * scale).tolist()
            item = {
                "bbox": {"x": x1, "y": y1, "width": max(x2 - x1, 1.0), "height": max(y2 - y1, 1.0)},
                "tip": {"x": tip_points[0][0], "y": tip_points[0][1]},
                "score": float(scores[b, q]),
                "tip2_score": float(exist[b, q]),
            }
            if exist[b, q] > 0.5:
                (ax, ay), (bx, by) = tip_points
                if math.hypot(ax - bx, ay - by) <= TIP_MERGE_PX:
                    item["tip"] = {"x": (ax + bx) / 2, "y": (ay + by) / 2}
                else:
                    item["tip2"] = {"x": bx, "y": by}
            annotations.append(item)
        results.append(annotations)
    return results
