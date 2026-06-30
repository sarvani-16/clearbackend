"""Loss utilities for Pix2Pix GAN."""

from __future__ import annotations

import torch
import torch.nn as nn


class Pix2PixLoss:
    """Pix2Pix loss with BCE adversarial term + weighted L1 reconstruction."""

    def __init__(self, lambda_l1: float = 100.0) -> None:
        self.lambda_l1 = lambda_l1
        self.gan_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def discriminator_loss(
        self,
        disc_real_logits: torch.Tensor,
        disc_fake_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_targets = torch.ones_like(disc_real_logits)
        fake_targets = torch.zeros_like(disc_fake_logits)
        d_real = self.gan_loss(disc_real_logits, real_targets)
        d_fake = self.gan_loss(disc_fake_logits, fake_targets)
        d_total = 0.5 * (d_real + d_fake)
        return d_total, d_real, d_fake

    def generator_loss(
        self,
        disc_fake_logits: torch.Tensor,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_targets = torch.ones_like(disc_fake_logits)
        g_gan = self.gan_loss(disc_fake_logits, real_targets)
        g_l1 = self.l1_loss(generated, target)
        g_total = g_gan + self.lambda_l1 * g_l1
        return g_total, g_gan, g_l1

