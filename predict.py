"""Inference script for cloud removal using trained Pix2Pix generator."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from models.generator import UNetGenerator
from utils.checkpoint import load_checkpoint


LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pix2Pix inference on cloudy images.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/patches/test/cloudy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def preprocess(image: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image * 2.0) - 1.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float().unsqueeze(0)


def postprocess(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.squeeze(0).detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip((image + 1.0) / 2.0, 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = Config()
    device = cfg.runtime.get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in args.input_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
    if not images:
        raise ValueError(f"No images found in {args.input_dir}")

    generator = UNetGenerator(in_channels=cfg.data.num_channels, out_channels=cfg.data.num_channels).to(device)
    load_checkpoint(path=args.checkpoint, generator=generator, map_location=device)
    generator.eval()

    with torch.no_grad():
        for img_path in tqdm(images, desc="Predict"):
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                LOGGER.warning("Skipping unreadable image: %s", img_path)
                continue
            tensor = preprocess(image).to(device)
            pred = generator(tensor)
            out = postprocess(pred)
            out_path = args.output_dir / f"{img_path.stem}_clear.png"
            cv2.imwrite(str(out_path), out)
    LOGGER.info("Saved predictions to %s", args.output_dir)


if __name__ == "__main__":
    main()

