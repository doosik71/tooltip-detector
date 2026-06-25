import torch
import torch.nn as nn
import torch.nn.functional as F


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Conv block with Squeeze-and-Excitation."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        mid = in_channels * expand_ratio
        se_ch = max(1, int(in_channels * se_ratio))
        self._skip = stride == 1 and in_channels == out_channels

        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels, mid, 1, bias=False)
            self._bn0 = nn.BatchNorm2d(mid)

        self._depthwise_conv = nn.Conv2d(
            mid, mid, kernel_size, stride=stride, padding=kernel_size // 2, groups=mid, bias=False
        )
        self._bn1 = nn.BatchNorm2d(mid)
        self._se_reduce = nn.Conv2d(mid, se_ch, 1)
        self._se_expand = nn.Conv2d(se_ch, mid, 1)
        self._project_conv = nn.Conv2d(mid, out_channels, 1, bias=False)
        self._bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        h = x
        if hasattr(self, "_expand_conv"):
            h = F.silu(self._bn0(self._expand_conv(h)))
        h = F.silu(self._bn1(self._depthwise_conv(h)))
        se = F.adaptive_avg_pool2d(h, 1)
        se = F.silu(self._se_reduce(se))
        se = torch.sigmoid(self._se_expand(se))
        h = h * se
        h = self._bn2(self._project_conv(h))
        return h + x if self._skip else h


class _BlockStage(nn.Module):
    """Ordered container for MBConv blocks within a stage.

    Uses global block indices as module names so the state_dict matches
    the pattern encoder._blocks.<stage_idx>.<global_block_idx>.*.
    """

    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x


class EfficientNetB2Encoder(nn.Module):
    """EfficientNet-B2 encoder (width_coeff=1.1, depth_coeff=1.2).

    Channel widths per stage: 16, 24, 48, 88, 120, 208, 352 → head 1408.
    Block counts per stage:    2,  3,  3,  4,  4,   5,   2.
    """

    # (in_ch, out_ch, kernel, stride, expand_ratio, num_blocks)
    _STAGES = [
        (32,  16, 3, 1, 1, 2),
        (16,  24, 3, 2, 6, 3),
        (24,  48, 5, 2, 6, 3),
        (48,  88, 3, 2, 6, 4),
        (88, 120, 5, 1, 6, 4),
        (120, 208, 5, 2, 6, 5),
        (208, 352, 3, 1, 6, 2),
    ]

    def __init__(self):
        super().__init__()
        self._conv_stem = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self._bn0 = nn.BatchNorm2d(32)

        self._blocks = nn.ModuleList()
        idx = 0
        for in_ch, out_ch, k, s, e, n in self._STAGES:
            stage = _BlockStage()
            for i in range(n):
                block = MBConvBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    k,
                    stride=s if i == 0 else 1,
                    expand_ratio=e,
                )
                stage.add_module(str(idx), block)
                idx += 1
            self._blocks.append(stage)

        self._conv_head = nn.Conv2d(352, 1408, 1, bias=False)
        self._bn1 = nn.BatchNorm2d(1408)
        self._fc = nn.Linear(1408, 1000)

    def extract_features(self, x):
        """Return feature maps at skip-connection stages.

        Returns a list of five tensors:
          [s0: 16ch H/2, s1: 24ch H/4, s2: 48ch H/8, s4: 120ch H/16, s6: 352ch H/32]
        """
        x = F.silu(self._bn0(self._conv_stem(x)))
        features = []
        for i, stage in enumerate(self._blocks):
            x = stage(x)
            if i in (0, 1, 2, 4, 6):
                features.append(x)
        return features

    def forward(self, x):
        x = F.silu(self._bn0(self._conv_stem(x)))
        for stage in self._blocks:
            x = stage(x)
        x = F.silu(self._bn1(self._conv_head(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self._fc(x)


class ADN(nn.Sequential):
    """Single-normalization ADN module (MONAI-compatible naming: adn.N)."""

    def __init__(self, channels):
        super().__init__()
        self.add_module("N", nn.BatchNorm2d(channels))


class _ConvBnBlock(nn.Module):
    """Conv2d + BatchNorm with MONAI-compatible sub-module names (conv / adn.N)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.adn = ADN(out_channels)

    def forward(self, x):
        return F.relu(self.adn(self.conv(x)), inplace=True)


class _DecoderConvs(nn.Module):
    """Two consecutive ConvBn blocks for a decoder stage."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_0 = _ConvBnBlock(in_channels, out_channels)
        self.conv_1 = _ConvBnBlock(out_channels, out_channels)

    def forward(self, x):
        return self.conv_1(self.conv_0(x))


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = _DecoderConvs(in_channels, out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.convs(x)


class UNetDecoder(nn.Module):
    """U-Net decoder that consumes EfficientNet-B2 feature maps.

    Skip connections (from extract_features output):
      block 0: upsample s6(352) → concat s4(120) → in 472, out 256
      block 1: upsample     256 → concat s2( 48) → in 304, out 128
      block 2: upsample     128 → concat s1( 24) → in 152, out  64
      block 3: upsample      64 → concat s0( 16) → in  80, out  32
      block 4: upsample      32 → (no concat)    → in  32, out  16
    """

    _BLOCK_CFG = [(472, 256), (304, 128), (152, 64), (80, 32), (32, 16)]

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_DecoderBlock(in_ch, out_ch) for in_ch, out_ch in self._BLOCK_CFG]
        )

    def forward(self, features):
        # features = [s0, s1, s2, s4, s6]
        bottleneck = features[4]
        skip_order = [features[3], features[2], features[1], features[0], None]
        x = bottleneck
        for block, skip in zip(self.blocks, skip_order):
            x = block(x, skip)
        return x


class TooltipDetector(nn.Module):
    """Surgical tool segmentation: EfficientNet-B2 encoder + U-Net decoder."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = EfficientNetB2Encoder()
        self.decoder = UNetDecoder()
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(16, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        features = self.encoder.extract_features(x)
        x = self.decoder(features)
        return self.segmentation_head(x)
