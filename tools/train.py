"""
Training Script — OmniPerceive
================================
Usage:
    python tools/train.py --config configs/kitti_multitask.yaml
    python tools/train.py --config configs/nuscenes_multitask.yaml --resume checkpoints/last.pth

Features:
  - Mixed-precision (torch.cuda.amp) training
  - Gradient clipping
  - OneCycleLR scheduler
  - TensorBoard logging of all task losses + log_sigma uncertainty weights
  - Distributed Data Parallel (DDP) via torchrun
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from models.omniperceive import OmniPerceive
from datasets import build_dataloader
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import get_logger
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="OmniPerceive Training")
    p.add_argument("--config",  required=True,  help="Path to YAML config")
    p.add_argument("--resume",  default=None,   help="Checkpoint to resume from")
    p.add_argument("--local_rank", type=int, default=0)
    return p.parse_args()


def setup_ddp(local_rank: int):
    """Initialise process group for multi-GPU training."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return True
    return False


def train_one_epoch(
    model, loader, optimizer, scaler, scheduler,
    epoch, cfg, writer, logger, device, is_main
):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(loader):
        images  = batch["image"].to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                   for k, v in batch["targets"].items()}

        optimizer.zero_grad(set_to_none=True)

        with autocast():                         # mixed-precision forward
            loss_dict = model(images, targets)
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            model.parameters(), cfg["training"]["gradient_clip"]
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        if is_main and step % cfg["logging"]["log_every"] == 0:
            global_step = epoch * len(loader) + step
            lr = scheduler.get_last_lr()[0]

            writer.add_scalar("train/loss_total",  loss_dict["loss"].item(),       global_step)
            writer.add_scalar("train/loss_det",    loss_dict["loss_det"].item(),   global_step)
            writer.add_scalar("train/loss_lane",   loss_dict["loss_lane"].item(),  global_step)
            writer.add_scalar("train/loss_depth",  loss_dict["loss_depth"].item(), global_step)
            writer.add_scalar("train/loss_seg",    loss_dict["loss_seg"].item(),   global_step)
            writer.add_scalar("train/lr",          lr,                             global_step)

            # Log per-task uncertainty weights (key insight of the paper)
            log_vars = loss_dict["log_vars"]
            for i, name in enumerate(["det", "lane", "depth", "seg"]):
                writer.add_scalar(f"uncertainty/log_sigma_{name}", log_vars[i].item(), global_step)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch [{epoch}] Step [{step}/{len(loader)}] "
                f"Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                f"sigma: {log_vars.exp().tolist()} | {elapsed:.1f}s"
            )
            t0 = time.time()

    return total_loss / len(loader)


def main():
    args = parse_args()
    is_ddp = setup_ddp(args.local_rank)
    is_main = (not is_ddp) or (dist.get_rank() == 0)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    logger = get_logger("omniperceive", cfg["logging"]["log_dir"])

    # ── Model ─────────────────────────────────────────────────────────────
    model = OmniPerceive(cfg["model"]).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloader(cfg, is_ddp)

    # ── Optimiser + Scheduler ─────────────────────────────────────────────
    opt_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=opt_cfg["betas"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=opt_cfg["lr"],
        steps_per_epoch=len(train_loader),
        epochs=cfg["training"]["epochs"],
        pct_start=cfg["training"]["scheduler"]["pct_start"],
    )
    scaler = GradScaler()

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, logger)

    writer = SummaryWriter(log_dir=cfg["logging"]["log_dir"]) if is_main else None

    # ── Training Loop ─────────────────────────────────────────────────────
    best_loss = float("inf")
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)

        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, scheduler,
            epoch, cfg, writer, logger, device, is_main
        )

        if is_main:
            logger.info(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

            if epoch % cfg["logging"]["save_every"] == 0 or avg_loss < best_loss:
                is_best = avg_loss < best_loss
                best_loss = min(avg_loss, best_loss)
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.module.state_dict() if is_ddp else model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_loss": best_loss,
                    },
                    save_dir=cfg["logging"]["save_dir"],
                    is_best=is_best,
                )

    if writer:
        writer.close()


if __name__ == "__main__":
    main()