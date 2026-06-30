"""Verify paired cloudy and clear images before patch generation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

if __package__:
    from .prepare_dataset import collect_cloudy_images, collect_image_pairs, read_image, validate_pair
else:
    from prepare_dataset import collect_cloudy_images, collect_image_pairs, read_image, validate_pair


LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify paired images or cloudy-only readiness.")
    parser.add_argument("--cloudy-dir", type=Path, default=Path("data/cloudy"))
    parser.add_argument("--clear-dir", type=Path, default=Path("data/clear"))
    parser.add_argument(
        "--mode",
        type=str,
        default="paired",
        choices=["paired", "single"],
        help="paired: expects data/clear. single: validates cloudy images only.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    failures = 0

    if args.mode == "paired":
        pairs = collect_image_pairs(args.cloudy_dir, args.clear_dir, exts)
        for cloudy_path, clear_path in pairs:
            try:
                cloudy = read_image(cloudy_path)
                clear = read_image(clear_path)
                validate_pair(cloudy, clear, cloudy_path.stem)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                LOGGER.error("Pair failed [%s]: %s", cloudy_path.stem, exc)
        if failures > 0:
            raise RuntimeError(f"Pair verification failed for {failures} image pairs.")
        LOGGER.info("All %d pairs passed validation.", len(pairs))
        return

    # single mode
    cloudy_images = collect_cloudy_images(args.cloudy_dir, exts)
    failures = 0
    if failures > 0:
        raise RuntimeError(f"Cloudy verification failed for {failures} images.")

    for cloudy_path in cloudy_images:
        try:
            cloudy = read_image(cloudy_path)
            validate_pair(cloudy, cloudy, cloudy_path.stem)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            LOGGER.error("Cloudy image failed [%s]: %s", cloudy_path.stem, exc)

    if failures > 0:
        raise RuntimeError(f"Cloudy verification failed for {failures} images.")
    LOGGER.info("All %d cloudy images passed validation.", len(cloudy_images))


if __name__ == "__main__":
    main()

