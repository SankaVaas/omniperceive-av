"""
Checkpoint Utilities
=====================
Save / load model checkpoints with full training state.
Supports:
  - Best-model tracking (saves best.pth separately)
  - Safe loading: missing keys are warned, unexpected keys are skipped
  - Colab/Kaggle: auto-copies best.pth to Google Drive if mounted
"""

import os
import shutil
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn


def save_checkpoint(
    state:    dict,
    save_dir: str,
    is_best:  bool = False,
    filename: str  = "last.pth",
) -> None:
    """
    Save training state dict to save_dir/filename.
    If is_best, also copy to save_dir/best.pth.

    state dict should contain:
        epoch       : int
        state_dict  : model weights (model.state_dict())
        optimizer   : optimizer.state_dict()
        scheduler   : scheduler.state_dict()
        best_loss   : float
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = Path(save_dir) / filename
    torch.save(state, str(path))

    if is_best:
        best_path = Path(save_dir) / "best.pth"
        shutil.copyfile(str(path), str(best_path))

        # Colab convenience: copy to Google Drive if mounted
        gdrive = Path("/content/drive/MyDrive/omniperceive_checkpoints")
        if gdrive.parent.exists():
            gdrive.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(best_path), str(gdrive / "best.pth"))


def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: Optional[object] = None,
    scheduler: Optional[object] = None,
    logger     = None,
) -> int:
    """
    Load checkpoint from path into model (and optionally optimizer/scheduler).

    Args:
        path      : path to .pth file.
        model     : model to load weights into (handles DDP wrapper).
        optimizer : if provided, restore optimizer state.
        scheduler : if provided, restore scheduler state.
        logger    : optional logger for messages.

    Returns:
        start_epoch : epoch to resume from (state["epoch"]).
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    if not os.path.isfile(path):
        log(f"[Checkpoint] No checkpoint found at '{path}' — starting from scratch.")
        return 0

    log(f"[Checkpoint] Loading '{path}' …")
    ckpt = torch.load(path, map_location="cpu")

    # Handle DDP-wrapped model (state_dict keys may have "module." prefix)
    raw_model = model.module if hasattr(model, "module") else model
    state = ckpt.get("state_dict", ckpt)

    missing, unexpected = raw_model.load_state_dict(state, strict=False)
    if missing:
        log(f"[Checkpoint] Missing keys ({len(missing)}): {missing[:5]} …")
    if unexpected:
        log(f"[Checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")

    if optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            log("[Checkpoint] Optimizer state restored.")
        except Exception as e:
            log(f"[Checkpoint] Could not restore optimizer: {e}")

    if scheduler and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
            log("[Checkpoint] Scheduler state restored.")
        except Exception as e:
            log(f"[Checkpoint] Could not restore scheduler: {e}")

    epoch = ckpt.get("epoch", 0)
    loss  = ckpt.get("best_loss", float("inf"))
    log(f"[Checkpoint] Resumed from epoch {epoch} | best_loss={loss:.4f}")
    return epoch