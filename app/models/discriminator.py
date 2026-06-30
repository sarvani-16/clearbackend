"""PatchGAN discriminator for Pix2Pix cloud removal."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiscBlock(nn.Module):
    """PatchGAN block: Conv -> (BN) -> LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, use_bn: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=not use_bn,
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(self, in_channels: int = 3, features: list[int] | None = None) -> None:
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.initial = DiscBlock(in_channels * 2, features[0], use_bn=False)
        layers: list[nn.Module] = []
        in_c = features[0]
        for idx, feat in enumerate(features[1:]):
            stride = 1 if idx == len(features[1:]) - 1 else 2
            layers.append(DiscBlock(in_c, feat, stride=stride, use_bn=True))
            in_c = feat
        layers.append(nn.Conv2d(in_c, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, cloudy: torch.Tensor, clear_or_fake: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cloudy, clear_or_fake], dim=1)
        x = self.initial(x)
        return self.model(x)

