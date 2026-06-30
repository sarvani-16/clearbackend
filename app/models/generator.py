"""U-Net generator for Pix2Pix cloud removal."""

from __future__ import annotations

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """Encoder block: Conv -> (BN) -> LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Decoder block: Deconv -> BN -> ReLU -> (Dropout)."""

    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    """Pix2Pix U-Net generator for 256x256 images."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, features: int = 64) -> None:
        super().__init__()
        self.down1 = DownBlock(in_channels, features, use_batchnorm=False)
        self.down2 = DownBlock(features, features * 2)
        self.down3 = DownBlock(features * 2, features * 4)
        self.down4 = DownBlock(features * 4, features * 8)
        self.down5 = DownBlock(features * 8, features * 8)
        self.down6 = DownBlock(features * 8, features * 8)
        self.down7 = DownBlock(features * 8, features * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.up1 = UpBlock(features * 8, features * 8, use_dropout=True)
        self.up2 = UpBlock(features * 16, features * 8, use_dropout=True)
        self.up3 = UpBlock(features * 16, features * 8, use_dropout=True)
        self.up4 = UpBlock(features * 16, features * 8, use_dropout=False)
        self.up5 = UpBlock(features * 16, features * 4, use_dropout=False)
        self.up6 = UpBlock(features * 8, features * 2, use_dropout=False)
        self.up7 = UpBlock(features * 4, features, use_dropout=False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))
        return self.final(torch.cat([up7, d1], dim=1))

