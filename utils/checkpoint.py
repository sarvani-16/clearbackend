"""Checkpoint utilities for training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    path: Path,
    epoch: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: Optimizer,
    optimizer_d: Optimizer,
    scheduler_g: LRScheduler | None,
    scheduler_d: LRScheduler | None,
    best_val_l1: float,
) -> None:
    """Save complete training state."""
    payload: dict[str, Any] = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "best_val_l1": best_val_l1,
    }
    if scheduler_g is not None:
        payload["scheduler_g_state_dict"] = scheduler_g.state_dict()
    if scheduler_d is not None:
        payload["scheduler_d_state_dict"] = scheduler_d.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module | None = None,
    optimizer_g: Optimizer | None = None,
    optimizer_d: Optimizer | None = None,
    scheduler_g: LRScheduler | None = None,
    scheduler_d: LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, float]:
    """Load checkpoint and return (start_epoch, best_val_l1)."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["generator_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["state_dict"])
    else:
        generator.load_state_dict(checkpoint)

    if discriminator is not None and "discriminator_state_dict" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    if optimizer_g is not None and "optimizer_g_state_dict" in checkpoint:
        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    if optimizer_d is not None and "optimizer_d_state_dict" in checkpoint:
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
    if scheduler_g is not None and "scheduler_g_state_dict" in checkpoint:
        scheduler_g.load_state_dict(checkpoint["scheduler_g_state_dict"])
    if scheduler_d is not None and "scheduler_d_state_dict" in checkpoint:
        scheduler_d.load_state_dict(checkpoint["scheduler_d_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_l1 = float(checkpoint.get("best_val_l1", float("inf")))
    return start_epoch, best_val_l1

