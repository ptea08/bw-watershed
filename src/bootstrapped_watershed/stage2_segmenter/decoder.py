"""U-Net decoder attached to the frozen DINOv3 feature hierarchy.

Paper, Sec. 3.1: the retained pseudo-labeled crops supervise a U-Net decoder
attached to the frozen DINOv3 feature hierarchy. Multiscale backbone features
are passed to the corresponding decoder stages through skip connections, while
only the decoder parameters are updated.

This is the same frozen ConvNeXt-Small checkpoint stage 1 uses, but read at
all four spatial stages (strides 4/8/16/32) rather than at stage3 alone. They
plug into U-Net skip connections directly, and the stride-4 stage carries the
fine detail that decides where a one-pixel-wide boundary sits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import require_hf_token


class DoubleConv(nn.Sequential):
    """Standard U-Net block: (Conv-BN-ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    """One decoder stage: upsample, concatenate the encoder skip, convolve."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetDecoder(nn.Module):
    """U-Net decoder consuming the four ConvNeXt skip feature maps.

    Encoder stages run fine to coarse (strides 4, 8, 16, 32). The deepest one
    becomes the bottleneck; each decoder stage then upsamples and merges the
    next finer skip, and a final x4 upsample returns to input resolution.
    """

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32),
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        d0, d1, d2, d3 = decoder_channels

        self.bottleneck = DoubleConv(c4, d0)
        self.up3 = UpBlock(d0, c3, d1)
        self.up2 = UpBlock(d1, c2, d2)
        self.up1 = UpBlock(d2, c1, d3)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            DoubleConv(d3, d3),
        )
        self.head = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(d3, num_classes, kernel_size=1),
        )

    def forward(self, s1, s2, s3, s4, target_hw=None) -> torch.Tensor:
        """``s1..s4`` are encoder feature maps, fine to coarse."""
        x = self.bottleneck(s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.final_up(x)

        # The x4 upsample assumes the stride-4 skip was exactly H/4; correct it
        # if the input size made that untrue.
        if target_hw is not None and x.shape[-2:] != torch.Size(target_hw):
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return self.head(x)


class ConvNeXtUNet(nn.Module):
    """Frozen DINOv3-ConvNeXt-Small encoder + trainable U-Net decoder.

    Predicts three channels in ``data.class_names`` order: background,
    foreground, boundary. Output is at full input resolution.
    """

    def __init__(self, cfg, encoder_channels: list[int] | None = None):
        super().__init__()
        stage2 = cfg.stage2_segmenter
        self.frozen = bool(stage2.backbone.frozen)

        if encoder_channels is None:
            from transformers import DINOv3ConvNextBackbone

            self.backbone = DINOv3ConvNextBackbone.from_pretrained(
                stage2.backbone.name,
                out_features=list(stage2.backbone.out_features),
                token=require_hf_token(),
            )
            encoder_channels = list(self.backbone.channels)
        else:
            # Test / smoke-test path: skip the gated download and drive the
            # decoder from synthetic feature maps.
            self.backbone = None

        if self.backbone is not None and self.frozen:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
            self.backbone.eval()

        self.decode_head = UNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=tuple(stage2.decoder.channels),
            num_classes=cfg.data.num_classes,
            dropout=stage2.decoder.dropout,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                features = self.backbone(pixel_values)
        else:
            features = self.backbone(pixel_values)

        s1, s2, s3, s4 = features.feature_maps
        return self.decode_head(s1, s2, s3, s4, target_hw=pixel_values.shape[2:])

    def train(self, mode: bool = True):
        """Keep the backbone in eval mode even while training the decoder.

        Otherwise its normalisation layers would keep updating their running
        statistics, which would make a nominally frozen encoder drift.
        """
        super().train(mode)
        if self.backbone is not None and self.frozen:
            self.backbone.eval()
        return self

    def trainable_parameters(self):
        """Decoder parameters only — the encoder stays frozen."""
        return self.decode_head.parameters()
