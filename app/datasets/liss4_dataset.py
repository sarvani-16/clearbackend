"""PyTorch dataset and dataloader utilities for LISS-IV Pix2Pix training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from config import Config


def build_transforms(image_size: int, is_train: bool) -> A.Compose:
    """Build augmentation pipeline."""
    if is_train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.2),
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.05,
                    p=0.3,
                ),
            ],
            additional_targets={"target": "image", "mask": "mask"},
        )
    return A.Compose(
        [A.Resize(image_size, image_size)],
        additional_targets={"target": "image", "mask": "mask"},
    )


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    return mask


def _to_tensor(image: np.ndarray) -> Tensor:
    image = image.astype(np.float32) / 255.0
    image = (image * 2.0) - 1.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float()


def _mask_to_tensor(mask: np.ndarray) -> Tensor:
    mask = (mask.astype(np.float32) / 255.0)[None, :, :]
    return torch.from_numpy(mask).float()


class LISS4Pix2PixDataset(Dataset):
    """Paired cloudy-clear patch dataset for Pix2Pix."""

    def __init__(
        self,
        root_dir: Path,
        split: str,
        image_size: int,
        is_train: bool,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.split = split
        self.root_dir = root_dir
        self.cloudy_dir = root_dir / split / "cloudy"
        self.clear_dir = root_dir / split / "clear"
        self.masks_dir = root_dir / split / "masks"
        self.transforms = build_transforms(image_size=image_size, is_train=is_train)

        if not self.cloudy_dir.exists() or not self.clear_dir.exists() or not self.masks_dir.exists():
            raise FileNotFoundError(f"Split directories missing for '{split}' in {root_dir}")

        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"No samples found for split '{split}' in {root_dir}")

    def _collect_samples(self) -> List[Tuple[Path, Path, Path]]:
        cloudy_files = sorted(self.cloudy_dir.glob("*.png"))
        samples: List[Tuple[Path, Path, Path]] = []
        for cloudy_path in cloudy_files:
            clear_path = self.clear_dir / cloudy_path.name
            mask_path = self.masks_dir / cloudy_path.name
            if clear_path.exists() and mask_path.exists():
                samples.append((cloudy_path, clear_path, mask_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        cloudy_path, clear_path, mask_path = self.samples[index]
        cloudy = _read_rgb(cloudy_path)
        clear = _read_rgb(clear_path)
        mask = _read_mask(mask_path)

        augmented = self.transforms(image=cloudy, target=clear, mask=mask)
        cloudy = augmented["image"]
        clear = augmented["target"]
        mask = augmented["mask"]

        return {
            "cloudy": _to_tensor(cloudy),
            "clear": _to_tensor(clear),
            "mask": _mask_to_tensor(mask),
            "sample_id": cloudy_path.stem,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    """Build dataloader with recommended defaults for GAN training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )


def create_dataloaders(
    config: Config,
    patches_root: Path = Path("data/patches"),
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders."""
    train_dataset = LISS4Pix2PixDataset(
        root_dir=patches_root,
        split="train",
        image_size=config.data.image_size,
        is_train=True,
    )
    val_dataset = LISS4Pix2PixDataset(
        root_dir=patches_root,
        split="val",
        image_size=config.data.image_size,
        is_train=False,
    )
    test_dataset = LISS4Pix2PixDataset(
        root_dir=patches_root,
        split="test",
        image_size=config.data.image_size,
        is_train=False,
    )

    return {
        "train": create_dataloader(
            train_dataset,
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory,
            shuffle=True,
        ),
        "val": create_dataloader(
            val_dataset,
            batch_size=config.eval.batch_size,
            num_workers=config.eval.num_workers,
            pin_memory=config.train.pin_memory,
            shuffle=False,
        ),
        "test": create_dataloader(
            test_dataset,
            batch_size=config.eval.batch_size,
            num_workers=config.eval.num_workers,
            pin_memory=config.train.pin_memory,
            shuffle=False,
        ),
    }

