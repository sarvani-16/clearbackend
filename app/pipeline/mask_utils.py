"""Utilities for cloud mask preparation and pixel removal."""

import numpy as np
from PIL import Image


def to_binary_mask(mask_array: np.ndarray) -> np.ndarray:
    """Convert a grayscale mask to a binary uint8 array (0 or 255)."""
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    return np.where(mask_array > 127, 255, 0).astype(np.uint8)


def remove_cloud_pixels(original_rgb: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """
    Zero out cloud pixels in the original image.

    Pixels outside the cloud mask are preserved exactly.
    """
    masked = original_rgb.copy()
    cloud_pixels = binary_mask > 0
    masked[cloud_pixels] = 0
    return masked


def load_rgb_image(image_path: str) -> Image.Image:
    """Load an image file as RGB."""
    return Image.open(image_path).convert("RGB")


def load_grayscale_mask(mask_path: str) -> np.ndarray:
    """Load a mask image as a grayscale numpy array."""
    mask_image = Image.open(mask_path).convert("L")
    return np.array(mask_image)
