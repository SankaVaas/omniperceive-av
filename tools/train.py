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