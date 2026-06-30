"""Project-wide configuration for Pix2Pix cloud removal.

This module contains strongly-typed dataclass configurations and utility
helpers used across training, evaluation, and inference scripts.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch


@dataclass
class DataConfig:
    """Configuration for dataset paths and image handling."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(init=False)
    cloudy_dir: Path = field(init=False)
    clear_dir: Path = field(init=False)
    masks_dir: Path = field(init=False)
    image_size: int = 256
    num_channels: int = 3
    file_extensions: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.cloudy_dir = self.data_dir / "cloudy"
        self.clear_dir = self.data_dir / "clear"
        self.masks_dir = self.data_dir / "masks"


@dataclass
class TrainConfig:
    """Configuration for GAN training."""

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Core hyperparameters
    epochs: int = 200
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_l1: float = 100.0

    # Scheduler
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.5

    # Mixed precision
    use_amp: bool = True

    # Logging and saving
    log_interval: int = 50
    val_interval: int = 1
    save_interval: int = 1
    experiment_name: str = "pix2pix_liss4_cloud_removal"
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    tensorboard_dir: Path = field(default_factory=lambda: Path("runs"))

    # Resume
    resume_checkpoint: str | None = None


@dataclass
class EvalConfig:
    """Configuration for evaluation and inference."""

    batch_size: int = 4
    num_workers: int = 2
    save_predictions: bool = True
    prediction_output_dir: Path = field(default_factory=lambda: Path("outputs/predictions"))
    metrics_output_file: Path = field(default_factory=lambda: Path("outputs/metrics.json"))


@dataclass
class RuntimeConfig:
    """Runtime configuration for device and precision."""

    device_preference: Literal["auto", "cuda", "cpu"] = "auto"

    def get_device(self) -> torch.device:
        """Return the selected torch device."""
        if self.device_preference == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")
            return torch.device("cuda")

        if self.device_preference == "cpu":
            return torch.device("cpu")

        # Auto mode
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    """Top-level project config composed of all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def create_directories(self) -> None:
        """Create required project directories if they do not exist."""
        directories = [
            self.data.data_dir,
            self.data.cloudy_dir,
            self.data.clear_dir,
            self.data.masks_dir,
            self.train.checkpoint_dir,
            self.train.output_dir,
            self.train.tensorboard_dir,
            self.eval.prediction_output_dir,
        ]
        for path in directories:
            path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Seed value for all random number generators.
        deterministic: If True, enables deterministic torch behavior where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def is_amp_enabled(config: Config) -> bool:
    """Return whether mixed precision should be enabled."""
    device = config.runtime.get_device()
    return config.train.use_amp and device.type == "cuda"

