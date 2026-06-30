"""Image quality metrics for cloud-removal evaluation."""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def denormalize_tensor(image: torch.Tensor) -> torch.Tensor:
    """Convert image tensor from [-1,1] to [0,1]."""
    return torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)


def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor [0,1] to HWC float32 numpy."""
    image = image.detach().cpu().numpy()
    return np.transpose(image, (1, 2, 0)).astype(np.float32)


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(target, pred, data_range=1.0))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return float(structural_similarity(target, pred, data_range=1.0, channel_axis=2))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def batch_metrics(generated: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute average PSNR/SSIM/MAE for a batch."""
    generated = denormalize_tensor(generated)
    target = denormalize_tensor(target)

    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    mae_vals: list[float] = []
    for idx in range(generated.shape[0]):
        pred_np = tensor_to_numpy(generated[idx])
        tgt_np = tensor_to_numpy(target[idx])
        psnr_vals.append(compute_psnr(pred_np, tgt_np))
        ssim_vals.append(compute_ssim(pred_np, tgt_np))
        mae_vals.append(compute_mae(pred_np, tgt_np))

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "mae": float(np.mean(mae_vals)),
    }

