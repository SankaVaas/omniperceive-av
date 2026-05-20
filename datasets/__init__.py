"""
Dataset registry and dataloader builder.
Resolves dataset name from config and builds train/val DataLoaders.
"""

from torch.utils.data import DataLoader

from datasets.kitti_dataset    import KITTIMultiTaskDataset
from datasets.nuscenes_dataset import NuScenesMultiTaskDataset

DATASET_REGISTRY = {
    "KITTIMultiTask":    KITTIMultiTaskDataset,
    "nuScenesMultiTask": NuScenesMultiTaskDataset,
}


def build_dataset(cfg: dict, split: str):
    name = cfg["dataset"]["name"]
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name](cfg, split=split)


def build_dataloader(cfg: dict, use_ddp: bool = False):
    """
    Build train and val DataLoaders from config.

    Colab/Kaggle: num_workers is capped at 2 automatically when
    cfg.training.num_workers > 2 and available CPU count is low.
    """
    import os
    train_ds = build_dataset(cfg, split="train")
    val_ds   = build_dataset(cfg, split="val")

    # Cap workers on Colab/Kaggle (typically 2 vCPUs)
    max_workers = min(cfg["training"].get("num_workers", 4), os.cpu_count() or 2)

    train_sampler = None
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True)

    collate_fn = getattr(train_ds, "collate_fn", None)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=max_workers,
        pin_memory=cfg["training"].get("pin_memory", True),
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, cfg["training"]["batch_size"] // 2),
        shuffle=False,
        num_workers=max(1, max_workers // 2),
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader