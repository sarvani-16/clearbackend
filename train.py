"""Training script for Pix2Pix cloud removal on LISS-IV patches."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from config import Config, is_amp_enabled, set_global_seed
from datasets.liss4_dataset import create_dataloaders
from models.discriminator import PatchDiscriminator
from models.generator import UNetGenerator
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.losses import Pix2PixLoss
from utils.metrics import batch_metrics, denormalize_tensor


LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pix2Pix on LISS-IV paired patches.")
    parser.add_argument("--patches-root", type=Path, default=Path("data/patches"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    return parser.parse_args()


def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    criterion: Pix2PixLoss,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    generator.train()
    discriminator.train()
    running = {"g_total": 0.0, "g_gan": 0.0, "g_l1": 0.0, "d_total": 0.0}
    pbar = tqdm(train_loader, desc="Train", leave=False)
    for batch in pbar:
        cloudy = batch["cloudy"].to(device, non_blocking=True)
        clear = batch["clear"].to(device, non_blocking=True)

        optimizer_d.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            fake = generator(cloudy)
            d_real_logits = discriminator(cloudy, clear)
            d_fake_logits = discriminator(cloudy, fake.detach())
            d_total, _, _ = criterion.discriminator_loss(d_real_logits, d_fake_logits)
        scaler.scale(d_total).backward()
        scaler.step(optimizer_d)

        optimizer_g.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            d_fake_for_g = discriminator(cloudy, fake)
            g_total, g_gan, g_l1 = criterion.generator_loss(d_fake_for_g, fake, clear)
        scaler.scale(g_total).backward()
        scaler.step(optimizer_g)
        scaler.update()

        running["g_total"] += float(g_total.item())
        running["g_gan"] += float(g_gan.item())
        running["g_l1"] += float(g_l1.item())
        running["d_total"] += float(d_total.item())
        pbar.set_postfix(g=float(g_total.item()), d=float(d_total.item()))
    n = max(1, len(train_loader))
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def validate(
    generator: nn.Module,
    discriminator: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: Pix2PixLoss,
    device: torch.device,
) -> dict[str, float]:
    generator.eval()
    discriminator.eval()
    running = {"g_total": 0.0, "g_l1": 0.0, "d_total": 0.0, "psnr": 0.0, "ssim": 0.0, "mae": 0.0}
    for batch in tqdm(val_loader, desc="Val", leave=False):
        cloudy = batch["cloudy"].to(device, non_blocking=True)
        clear = batch["clear"].to(device, non_blocking=True)
        fake = generator(cloudy)
        d_real_logits = discriminator(cloudy, clear)
        d_fake_logits = discriminator(cloudy, fake)
        d_total, _, _ = criterion.discriminator_loss(d_real_logits, d_fake_logits)
        g_total, _, g_l1 = criterion.generator_loss(d_fake_logits, fake, clear)
        metrics = batch_metrics(fake, clear)
        running["g_total"] += float(g_total.item())
        running["g_l1"] += float(g_l1.item())
        running["d_total"] += float(d_total.item())
        running["psnr"] += metrics["psnr"]
        running["ssim"] += metrics["ssim"]
        running["mae"] += metrics["mae"]
    n = max(1, len(val_loader))
    return {k: v / n for k, v in running.items()}


def log_images(writer: SummaryWriter, cloudy: torch.Tensor, fake: torch.Tensor, clear: torch.Tensor, epoch: int) -> None:
    writer.add_image("samples/cloudy", make_grid(denormalize_tensor(cloudy[:4]), nrow=4), global_step=epoch)
    writer.add_image("samples/fake", make_grid(denormalize_tensor(fake[:4]), nrow=4), global_step=epoch)
    writer.add_image("samples/clear", make_grid(denormalize_tensor(clear[:4]), nrow=4), global_step=epoch)


def main() -> None:
    args = parse_args()
    setup_logging()
    config = Config()
    config.create_directories()
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.lr is not None:
        config.train.learning_rate = args.lr
    if args.resume is not None:
        config.train.resume_checkpoint = args.resume
    if args.exp_name is not None:
        config.train.experiment_name = args.exp_name

    set_global_seed(config.train.seed, config.train.deterministic)
    device = config.runtime.get_device()
    use_amp = is_amp_enabled(config)
    dataloaders = create_dataloaders(config=config, patches_root=args.patches_root)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    generator = UNetGenerator(in_channels=config.data.num_channels, out_channels=config.data.num_channels).to(device)
    discriminator = PatchDiscriminator(in_channels=config.data.num_channels).to(device)
    optimizer_g = Adam(generator.parameters(), lr=config.train.learning_rate, betas=(config.train.beta1, config.train.beta2))
    optimizer_d = Adam(discriminator.parameters(), lr=config.train.learning_rate, betas=(config.train.beta1, config.train.beta2))
    scheduler_g = StepLR(optimizer_g, step_size=config.train.scheduler_step_size, gamma=config.train.scheduler_gamma)
    scheduler_d = StepLR(optimizer_d, step_size=config.train.scheduler_step_size, gamma=config.train.scheduler_gamma)
    criterion = Pix2PixLoss(lambda_l1=config.train.lambda_l1)
    scaler = GradScaler(device=device.type, enabled=use_amp)
    writer = SummaryWriter(log_dir=str(config.train.tensorboard_dir / config.train.experiment_name))

    start_epoch = 1
    best_val_l1 = float("inf")
    if config.train.resume_checkpoint:
        start_epoch, best_val_l1 = load_checkpoint(
            path=Path(config.train.resume_checkpoint),
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
            map_location=device,
        )

    for epoch in range(start_epoch, config.train.epochs + 1):
        train_metrics = train_one_epoch(
            generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, scaler, device, use_amp
        )
        scheduler_g.step()
        scheduler_d.step()
        val_metrics = validate(generator, discriminator, val_loader, criterion, device)

        writer.add_scalar("train/g_total", train_metrics["g_total"], epoch)
        writer.add_scalar("train/g_gan", train_metrics["g_gan"], epoch)
        writer.add_scalar("train/g_l1", train_metrics["g_l1"], epoch)
        writer.add_scalar("train/d_total", train_metrics["d_total"], epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        sample_batch: dict[str, Any] = next(iter(val_loader))
        cloudy = sample_batch["cloudy"].to(device)
        clear = sample_batch["clear"].to(device)
        fake = generator(cloudy)
        log_images(writer, cloudy, fake, clear, epoch)

        latest_ckpt = config.train.checkpoint_dir / f"{config.train.experiment_name}_latest.pt"
        save_checkpoint(latest_ckpt, epoch, generator, discriminator, optimizer_g, optimizer_d, scheduler_g, scheduler_d, best_val_l1)
        if val_metrics["g_l1"] < best_val_l1:
            best_val_l1 = val_metrics["g_l1"]
            best_ckpt = config.train.checkpoint_dir / f"{config.train.experiment_name}_best.pt"
            save_checkpoint(best_ckpt, epoch, generator, discriminator, optimizer_g, optimizer_d, scheduler_g, scheduler_d, best_val_l1)

        LOGGER.info(
            "Epoch %d | Train G %.4f D %.4f | Val L1 %.4f PSNR %.2f SSIM %.4f MAE %.5f",
            epoch,
            train_metrics["g_total"],
            train_metrics["d_total"],
            val_metrics["g_l1"],
            val_metrics["psnr"],
            val_metrics["ssim"],
            val_metrics["mae"],
        )

    writer.close()
    LOGGER.info("Training complete.")


if __name__ == "__main__":
    main()

