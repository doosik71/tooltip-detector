"""Multi-scale deformable attention (Deformable DETR), in plain PyTorch.

The reference implementation ships a CUDA extension; this is its pure-PyTorch
fallback (`ms_deform_attn_core_pytorch`), built on `grid_sample`. It is slower
and uses more memory, but needs no compilation, which keeps the project
installable with the root `uv` environment alone.

The sampling runs in fp32 even under autocast: bilinear weights of nearby
sampling points differ by little, and half precision blurs them.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def ms_deform_attn_core(value: torch.Tensor, spatial_shapes: list[tuple[int, int]],
                        sampling_locations: torch.Tensor,
                        attention_weights: torch.Tensor) -> torch.Tensor:
    """value (B, S, heads, dim), sampling_locations (B, Q, heads, levels, points, 2)
    in [0, 1], attention_weights (B, Q, heads, levels, points) -> (B, Q, heads * dim)."""
    batch, _, heads, dim = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in spatial_shapes], dim=1)
    grids = 2 * sampling_locations - 1
    # (B, Q, heads, levels, points) -> (B*heads, 1, Q, levels, points)
    weights = attention_weights.transpose(1, 2).reshape(batch * heads, 1, queries, levels, points)
    output = 0
    for level, (h, w) in enumerate(spatial_shapes):
        # (B, H*W, heads, dim) -> (B*heads, dim, H, W)
        level_value = value_list[level].flatten(2).transpose(1, 2).reshape(batch * heads, dim, h, w)
        # (B, Q, heads, points, 2) -> (B*heads, Q, points, 2)
        level_grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled = F.grid_sample(level_value, level_grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False)    # (B*heads, dim, Q, P)
        # Summed level by level rather than stacked, which saves a large copy.
        output = output + (sampled * weights[:, :, :, level]).sum(-1)
    return output.view(batch, heads * dim, queries).transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    """Each query looks at `points` sampled locations per head and level,
    placed by offsets it predicts around its reference point (or box)."""

    def __init__(self, d_model: int = 256, levels: int = 3, heads: int = 8, points: int = 4):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.d_model, self.levels, self.heads, self.points = d_model, levels, heads, points
        self.sampling_offsets = nn.Linear(d_model, heads * levels * points * 2)
        self.attention_weights = nn.Linear(d_model, heads * levels * points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Initial offsets point outward in `heads` directions, growing with
        # the point index, as in the reference implementation.
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.heads, dtype=torch.float32) * (2.0 * math.pi / self.heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(self.heads, 1, 1, 2)
        grid = grid.repeat(1, self.levels, self.points, 1)
        for i in range(self.points):
            grid[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query: torch.Tensor, reference_points: torch.Tensor, value: torch.Tensor,
                spatial_shapes: list[tuple[int, int]],
                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """query (B, Q, C); reference_points (B, Q, levels, 2) points or
        (B, Q, levels, 4) boxes, normalised; value (B, S, C)."""
        batch, queries, _ = query.shape
        value = self.value_proj(value)
        if padding_mask is not None:
            value = value.masked_fill(padding_mask[..., None], 0.0)
        value = value.view(batch, -1, self.heads, self.d_model // self.heads)

        offsets = self.sampling_offsets(query).view(
            batch, queries, self.heads, self.levels, self.points, 2)
        weights = self.attention_weights(query).view(
            batch, queries, self.heads, self.levels * self.points)
        weights = weights.softmax(-1).view(batch, queries, self.heads, self.levels, self.points)

        if reference_points.shape[-1] == 2:
            # Offsets are in feature-map cells of each level.
            normaliser = torch.tensor([[w, h] for h, w in spatial_shapes],
                                      dtype=offsets.dtype, device=offsets.device)
            locations = (reference_points[:, :, None, :, None, :]
                         + offsets / normaliser[None, None, None, :, None, :])
        else:
            # Offsets scale with the reference box, half its size per unit.
            locations = (reference_points[:, :, None, :, None, :2]
                         + offsets / self.points * reference_points[:, :, None, :, None, 2:] * 0.5)

        with torch.autocast(device_type=query.device.type, enabled=False):
            output = ms_deform_attn_core(value.float(), spatial_shapes, locations.float(),
                                         weights.float())
        return self.output_proj(output.to(query.dtype))
