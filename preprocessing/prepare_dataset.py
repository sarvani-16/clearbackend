"""Dataset preparation pipeline for Pix2Pix cloud removal."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from tqdm import tqdm

from config import Config


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    split: str
    cloudy_path: str
    clear_path: str
    mask_path: str
    source_cloudy: str
    source_clear: str
    y: int
    x: int


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def collect_image_pairs(cloudy_dir: Path, clear_dir: Path, extensions: Sequence[str]) -> List[Tuple[Path, Path]]:
    if not cloudy_dir.exists():
        raise FileNotFoundError(f"Cloudy directory not found: {cloudy_dir}")
    if not clear_dir.exists():
        raise FileNotFoundError(f"Clear directory not found: {clear_dir}")

    ext_set = {ext.lower() for ext in extensions}
    cloudy_map: Dict[str, Path] = {}
    clear_map: Dict[str, Path] = {}

    for path in cloudy_dir.iterdir():
        if path.is_file() and path.suffix.lower() in ext_set:
            cloudy_map[path.stem] = path
    for path in clear_dir.iterdir():
        if path.is_file() and path.suffix.lower() in ext_set:
            clear_map[path.stem] = path

    common_stems = sorted(set(cloudy_map).intersection(clear_map))
    if not common_stems:
        raise ValueError("No paired files found. Ensure identical filenames in data/cloudy and data/clear.")

    pairs = [(cloudy_map[stem], clear_map[stem]) for stem in common_stems]
    LOGGER.info("Collected %d paired images.", len(pairs))
    return pairs


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    img = image.astype(np.float32)
    min_val = float(np.min(img))
    max_val = float(np.max(img))
    if max_val <= min_val:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - min_val) / (max_val - min_val)
    return np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)


def read_image(path: Path) -> np.ndarray:
    try:
        with rasterio.open(path) as src:
            arr = src.read()
        arr = np.transpose(arr, (1, 2, 0))
    except RasterioIOError:
        arr_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr_bgr is None:
            raise RuntimeError(f"Unable to read image: {path}") from None
        if arr_bgr.ndim == 2:
            arr = np.stack([arr_bgr] * 3, axis=-1)
        else:
            arr = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return to_uint8(arr)


def validate_pair(cloudy: np.ndarray, clear: np.ndarray, pair_id: str) -> None:
    if cloudy.shape != clear.shape:
        raise ValueError(f"Shape mismatch for pair '{pair_id}': cloudy={cloudy.shape}, clear={clear.shape}")
    if cloudy.ndim != 3 or cloudy.shape[2] != 3:
        raise ValueError(f"Expected 3-channel image for '{pair_id}', got shape {cloudy.shape}")


def generate_cloud_mask(cloudy_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cloudy_rgb, cv2.COLOR_RGB2HSV)
    _, s, v = cv2.split(hsv)
    cloud = np.logical_and(v > 180, s < 80).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    cloud = cv2.morphologyEx(cloud, cv2.MORPH_OPEN, kernel, iterations=1)
    cloud = cv2.morphologyEx(cloud, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cloud


def extract_patches(cloudy: np.ndarray, clear: np.ndarray, patch_size: int, stride: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
    height, width, _ = cloudy.shape
    if height < patch_size or width < patch_size:
        return []

    patches: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            cloudy_patch = cloudy[y : y + patch_size, x : x + patch_size]
            clear_patch = clear[y : y + patch_size, x : x + patch_size]
            mask_patch = generate_cloud_mask(cloudy_patch)
            patches.append((cloudy_patch, clear_patch, mask_patch, y, x))
    return patches


def inpaint_cloudy(cloudy_rgb: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
    """Create a pseudo-clear target by inpainting cloud pixels.

    Args:
        cloudy_rgb: RGB uint8 patch.
        cloud_mask: Binary-ish mask where cloud pixels are non-zero (uint8).

    Returns:
        RGB uint8 pseudo-clear patch.
    """
    # OpenCV expects a single-channel 8-bit mask where inpaint regions are 255.
    inpaint_mask = (cloud_mask > 0).astype(np.uint8) * 255
    cloudy_bgr = cv2.cvtColor(cloudy_rgb, cv2.COLOR_RGB2BGR)

    # Radius controls neighborhood size used for inpainting (small works for 256 patches).
    inpainted_bgr = cv2.inpaint(cloudy_bgr, inpaint_mask, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)


def extract_patches_single(cloudy: np.ndarray, patch_size: int, stride: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
    """Extract aligned patches when only cloudy images are available.

    The 'clear' target is synthesized using cloud masks + OpenCV inpainting.
    """
    height, width, _ = cloudy.shape
    if height < patch_size or width < patch_size:
        return []

    patches: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            cloudy_patch = cloudy[y : y + patch_size, x : x + patch_size]
            mask_patch = generate_cloud_mask(cloudy_patch)
            clear_patch = inpaint_cloudy(cloudy_patch, mask_patch)
            patches.append((cloudy_patch, clear_patch, mask_patch, y, x))
    return patches


def split_name(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_cutoff = int(total * train_ratio)
    val_cutoff = train_cutoff + int(total * val_ratio)
    if index < train_cutoff:
        return "train"
    if index < val_cutoff:
        return "val"
    return "test"


def ensure_output_dirs(base_dir: Path) -> None:
    for split in ("train", "val", "test"):
        for sub in ("cloudy", "clear", "masks"):
            (base_dir / split / sub).mkdir(parents=True, exist_ok=True)


def save_patch(path: Path, image: np.ndarray) -> None:
    if image.ndim == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(path), image_bgr)
    else:
        ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def collect_cloudy_images(cloudy_dir: Path, extensions: Sequence[str]) -> List[Path]:
    """Collect cloudy images for single-source mode."""
    if not cloudy_dir.exists():
        raise FileNotFoundError(f"Cloudy directory not found: {cloudy_dir}")
    ext_set = {ext.lower() for ext in extensions}
    images: List[Path] = []
    for path in cloudy_dir.iterdir():
        if path.is_file() and path.suffix.lower() in ext_set:
            images.append(path)
    images = sorted(images)
    if not images:
        raise ValueError(f"No images found in {cloudy_dir} with extensions {extensions}")
    LOGGER.info("Collected %d cloudy images for single-source mode.", len(images))
    return images


def get_image_hw(path: Path) -> tuple[int, int]:
    """Get (height, width) without loading full arrays (when possible)."""
    try:
        with rasterio.open(path) as src:
            return int(src.height), int(src.width)
    except RasterioIOError:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Unable to read image for shape: {path}")
        if img.ndim == 2:
            return int(img.shape[0]), int(img.shape[1])
        return int(img.shape[0]), int(img.shape[1])


def count_patches_for_hw(height: int, width: int, patch_size: int, stride: int) -> int:
    """Count number of sliding-window patches for given image dimensions."""
    if height < patch_size or width < patch_size:
        return 0
    n_y = (height - patch_size) // stride + 1
    n_x = (width - patch_size) // stride + 1
    return int(n_y * n_x)


def count_patches_for_path(path: Path, patch_size: int, stride: int) -> int:
    """Count number of patches for a single image path."""
    height, width = get_image_hw(path)
    return count_patches_for_hw(height=height, width=width, patch_size=patch_size, stride=stride)


def prepare_dataset(
    output_dir: Path,
    patch_size: int,
    stride: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    mode: Literal["paired", "single"] = "paired",
) -> None:
    cfg = Config()
    cfg.create_directories()
    ensure_output_dirs(output_dir)

    if mode == "paired":
        sources = collect_image_pairs(cfg.data.cloudy_dir, cfg.data.clear_dir, cfg.data.file_extensions)
    elif mode == "single":
        sources = collect_cloudy_images(cfg.data.cloudy_dir, cfg.data.file_extensions)
    else:  # pragma: no cover
        raise ValueError(f"Unknown mode: {mode}")

    rng = random.Random(seed)
    rng.shuffle(sources)

    # Split at patch-level (not image-level), so single-image projects still get train/val patches.
    total_patches = 0
    if mode == "paired":
        for cloudy_path, _clear_path in sources:  # type: ignore[misc]
            total_patches += count_patches_for_path(cloudy_path, patch_size=patch_size, stride=stride)
    else:
        for cloudy_path in sources:  # type: ignore[assignment]
            total_patches += count_patches_for_path(cloudy_path, patch_size=patch_size, stride=stride)

    if total_patches == 0:
        raise RuntimeError(
            "No patches were generated. Ensure your input image dimensions are >= patch-size "
            f"({patch_size}) and that --stride=({stride}) is valid."
        )

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1).")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0,1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    train_count = max(1, int(total_patches * train_ratio))
    val_count = int(total_patches * val_ratio)

    # Keep val non-empty when possible (training code logs a validation batch each epoch).
    if total_patches >= 2 and val_count == 0 and val_ratio > 0.0:
        val_count = 1
        train_count = max(1, train_count - 1)

    if train_count + val_count > total_patches:
        val_count = max(0, total_patches - train_count)

    train_cutoff = train_count
    val_cutoff = train_count + val_count

    manifest: List[PreparedSample] = []
    sample_counter = 0
    patch_idx = 0

    if mode == "paired":
        for _idx, (cloudy_path, clear_path) in enumerate(tqdm(sources, desc="Preparing pairs")):
            pair_name = cloudy_path.stem

            cloudy_img = read_image(cloudy_path)
            clear_img = read_image(clear_path)
            validate_pair(cloudy_img, clear_img, pair_name)

            patches = extract_patches(cloudy_img, clear_img, patch_size=patch_size, stride=stride)
            for cloudy_patch, clear_patch, mask_patch, y, x in patches:
                if patch_idx < train_cutoff:
                    split = "train"
                elif patch_idx < val_cutoff:
                    split = "val"
                else:
                    split = "test"
                sample_id = f"{pair_name}_{sample_counter:07d}"
                cloudy_out = output_dir / split / "cloudy" / f"{sample_id}.png"
                clear_out = output_dir / split / "clear" / f"{sample_id}.png"
                mask_out = output_dir / split / "masks" / f"{sample_id}.png"

                save_patch(cloudy_out, cloudy_patch)
                save_patch(clear_out, clear_patch)
                save_patch(mask_out, mask_patch)

                manifest.append(
                    PreparedSample(
                        sample_id=sample_id,
                        split=split,
                        cloudy_path=str(cloudy_out),
                        clear_path=str(clear_out),
                        mask_path=str(mask_out),
                        source_cloudy=str(cloudy_path),
                        source_clear=str(clear_path),
                        y=y,
                        x=x,
                    )
                )
                sample_counter += 1
                patch_idx += 1
    else:
        for _idx, cloudy_path in enumerate(tqdm(sources, desc="Preparing single-source cloudy images")):
            pair_name = cloudy_path.stem

            cloudy_img = read_image(cloudy_path)
            # Validate channel count/shape only.
            validate_pair(cloudy_img, cloudy_img, pair_name)

            patches = extract_patches_single(cloudy_img, patch_size=patch_size, stride=stride)
            for cloudy_patch, clear_patch, mask_patch, y, x in patches:
                if patch_idx < train_cutoff:
                    split = "train"
                elif patch_idx < val_cutoff:
                    split = "val"
                else:
                    split = "test"
                sample_id = f"{pair_name}_{sample_counter:07d}"
                cloudy_out = output_dir / split / "cloudy" / f"{sample_id}.png"
                clear_out = output_dir / split / "clear" / f"{sample_id}.png"
                mask_out = output_dir / split / "masks" / f"{sample_id}.png"

                save_patch(cloudy_out, cloudy_patch)
                save_patch(clear_out, clear_patch)
                save_patch(mask_out, mask_patch)

                manifest.append(
                    PreparedSample(
                        sample_id=sample_id,
                        split=split,
                        cloudy_path=str(cloudy_out),
                        clear_path=str(clear_out),
                        mask_path=str(mask_out),
                        source_cloudy=str(cloudy_path),
                        source_clear="generated_via_inpainting",
                        y=y,
                        x=x,
                    )
                )
                sample_counter += 1
                patch_idx += 1

    if not manifest:
        raise RuntimeError(
            "No patches were generated. Check that your input image is larger than --patch-size "
            f"({patch_size}) and that stride (--stride={stride}) is valid. Also verify cloud masks are generated."
        )

    (output_dir / "manifest.json").write_text(
        json.dumps([m.__dict__ for m in manifest], indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Saved %d patches.", len(manifest))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare patch dataset for Pix2Pix training.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/patches"))
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        default="paired",
        choices=["paired", "single"],
        help="Use 'paired' if data/clear exists, or 'single' to generate pseudo-clear via inpainting.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    prepare_dataset(
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        mode=args.mode,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()

