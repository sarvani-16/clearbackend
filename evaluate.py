"""Evaluate Pix2Pix generator on test split with PSNR/SSIM/MAE."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from config import Config
from datasets.liss4_dataset import create_dataloaders
from models.generator import UNetGenerator
from utils.checkpoint import load_checkpoint
from utils.metrics import batch_metrics, denormalize_tensor


LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pix2Pix checkpoint on test split.")
    parser.add_argument("--patches-root", type=Path, default=Path("data/patches"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    return parser.parse_args()


def save_batch_predictions(cloudy: torch.Tensor, generated: torch.Tensor, clear: torch.Tensor, sample_ids: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cloudy = denormalize_tensor(cloudy)
    generated = denormalize_tensor(generated)
    clear = denormalize_tensor(clear)
    for idx, sample_id in enumerate(sample_ids):
        c = (cloudy[idx].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        g = (generated[idx].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        t = (clear[idx].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        cv2.imwrite(str(output_dir / f"{sample_id}_cloudy.png"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / f"{sample_id}_pred.png"), cv2.cvtColor(g, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / f"{sample_id}_clear.png"), cv2.cvtColor(t, cv2.COLOR_RGB2BGR))


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = Config()
    cfg.create_directories()
    device = cfg.runtime.get_device()

    test_loader = create_dataloaders(config=cfg, patches_root=args.patches_root)["test"]
    generator = UNetGenerator(in_channels=cfg.data.num_channels, out_channels=cfg.data.num_channels).to(device)
    load_checkpoint(path=args.checkpoint, generator=generator, map_location=device)
    generator.eval()

    totals = {"psnr": 0.0, "ssim": 0.0, "mae": 0.0}
    count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluate"):
            cloudy = batch["cloudy"].to(device, non_blocking=True)
            clear = batch["clear"].to(device, non_blocking=True)
            generated = generator(cloudy)
            metrics = batch_metrics(generated, clear)
            for key in totals:
                totals[key] += metrics[key]
            count += 1
            if args.save_images:
                save_batch_predictions(cloudy, generated, clear, list(batch["sample_id"]), args.output_dir)

    if count == 0:
        raise RuntimeError("No batches found in test loader.")

    results = {key: value / count for key, value in totals.items()}
    metrics_path = cfg.eval.metrics_output_file
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Test Metrics: PSNR=%.3f | SSIM=%.4f | MAE=%.5f", results["psnr"], results["ssim"], results["mae"])


if __name__ == "__main__":
    main()

