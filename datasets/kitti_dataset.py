"""
KITTI Multi-Task Dataset
========================
Loads all four task targets from a standard KITTI layout:

  data/kitti/
    training/
      image_2/          ← left RGB (stereo left)
      velodyne/         ← LiDAR point clouds (.bin, float32 xyzr)
      label_2/          ← 3D box annotations (KITTI format)
      calib/            ← calibration files per scene
      depth_annotated/  ← sparse GT depth PNGs (eval only — NOT used in loss)
      gt_image_2/       ← KITTI Road segmentation masks (drivable area)
    ImageSets/
      train.txt         ← 6-digit frame indices, one per line
      val.txt

For depth self-supervision, consecutive frames are loaded as source views.
Velodyne sweeps are projected to image space to provide a LiDAR depth map
for evaluation (not training signal).

Usage:
    from datasets.kitti_dataset import KITTIMultiTaskDataset
    ds = KITTIMultiTaskDataset(cfg, split='train')
    sample = ds[0]
    # sample.keys() → image, depth, seg_mask, targets, calib, meta
"""

import os
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset

from datasets.transforms import build_transforms
from datasets.utils import (
    load_kitti_depth,
    build_depth_map_from_lidar,
    render_heatmap,
    box3d_to_bev,
)


# ── KITTI class mapping ────────────────────────────────────────────────────
KITTI_CLASSES = {
    "Car":        0,
    "Pedestrian": 1,
    "Cyclist":    2,
}


class KITTICalib:
    """Parse and cache a KITTI calibration file."""

    def __init__(self, path: str):
        data = {}
        with open(path) as f:
            for line in f:
                if ":" not in line:
                    continue
                key, val = line.strip().split(":", 1)
                data[key] = np.array([float(x) for x in val.split()])

        self.P2  = data["P2"].reshape(3, 4).astype(np.float32)
        self.R0  = data["R0_rect"].reshape(3, 3).astype(np.float32)
        self.Tr  = data["Tr_velo_to_cam"].reshape(3, 4).astype(np.float32)

        # Extend Tr to 4×4 for matrix multiplication
        self.Tr_hom = np.vstack([self.Tr, [0, 0, 0, 1]]).astype(np.float32)

    def as_dict(self):
        return {"P2": self.P2, "R0": self.R0, "Tr": self.Tr}


class KITTILabel:
    """Parse a KITTI label_2 file into numpy arrays."""

    __slots__ = ["type", "truncated", "occluded", "alpha",
                 "bbox", "dimensions", "location", "rotation_y"]

    def __init__(self, line: str):
        parts = line.strip().split()
        self.type       = parts[0]
        self.truncated  = float(parts[1])
        self.occluded   = int(parts[2])
        self.alpha      = float(parts[3])
        self.bbox       = np.array([float(x) for x in parts[4:8]])   # 2D box [x1,y1,x2,y2]
        self.dimensions = np.array([float(x) for x in parts[8:11]])  # h, w, l
        self.location   = np.array([float(x) for x in parts[11:14]]) # x, y, z in cam
        self.rotation_y = float(parts[14])


def parse_labels(label_path: str, class_map: dict):
    """
    Parse KITTI label file → boxes_3d (N,7) in LiDAR frame, class_ids (N,).
    Boxes are converted from camera coords [x,y,z,h,w,l,ry] to LiDAR [x,y,z,l,w,h,yaw].
    """
    if not os.path.exists(label_path):
        return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.int64)

    boxes, classes = [], []
    with open(label_path) as f:
        for line in f:
            lbl = KITTILabel(line)
            if lbl.type not in class_map:
                continue
            if lbl.occluded > 2 or lbl.truncated > 0.5:
                continue   # skip heavily occluded / truncated

            h, w, l = lbl.dimensions
            x, y, z = lbl.location           # camera frame

            # Camera → LiDAR approximate conversion (KITTI convention)
            # LiDAR x=forward, y=left, z=up; Camera x=right, y=down, z=forward
            lx =  z
            ly = -x
            lz = -y + h / 2.0               # bottom → center

            yaw = -(lbl.rotation_y + np.pi / 2)  # camera ry → LiDAR yaw

            boxes.append([lx, ly, lz, l, w, h, yaw])
            classes.append(class_map[lbl.type])

    if not boxes:
        return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.int64)

    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int64)


class KITTIMultiTaskDataset(Dataset):
    """
    PyTorch Dataset for KITTI multi-task learning.

    Returns per sample:
        image       : (3, H, W) float32 tensor, normalized
        depth       : (1, H, W) float32 — LiDAR-projected sparse depth (eval only)
        seg_mask    : (1, H, W) uint8   — drivable area binary mask
        source_imgs : list of (3, H, W) tensors for depth self-supervision
        targets     : dict with heatmap, offset, wh, class_ids, boxes_3d, lane_*
        calib       : dict with P2, R0, Tr matrices
        meta        : frame_id, split, img_path

    Args:
        cfg   : full YAML config dict.
        split : 'train' or 'val'.
    """

    def __init__(self, cfg: dict, split: str = "train"):
        self.cfg     = cfg
        self.split   = split
        self.root    = Path(cfg["dataset"]["root"])
        self.img_dir = self.root / "training" / "image_2"
        self.vel_dir = self.root / "training" / "velodyne"
        self.lbl_dir = self.root / "training" / "label_2"
        self.cal_dir = self.root / "training" / "calib"
        self.dep_dir = self.root / "training" / "depth_annotated"
        self.seg_dir = self.root / "training" / "gt_image_2"

        split_file = self.root / "ImageSets" / f"{split}.txt"
        with open(split_file) as f:
            self.frame_ids = [line.strip() for line in f if line.strip()]

        self.transforms  = build_transforms(cfg["training"], split)
        self.class_map   = {c: i for c, i in zip(
            cfg["dataset"]["class_names"], cfg["dataset"]["class_ids"]
        )}
        self.num_classes = len(self.class_map)
        self.bev_cfg     = {
            "pc_range": cfg["model"]["bev_neck"]["pc_range"],
            "bev_h":    cfg["model"]["bev_neck"]["bev_h"],
            "bev_w":    cfg["model"]["bev_neck"]["bev_w"],
        }
        self.frame_offsets = cfg["dataset"].get("frame_offsets", [-1, 0, 1])

    def __len__(self):
        return len(self.frame_ids)

    def _load_image(self, frame_id: str) -> np.ndarray:
        path = self.img_dir / f"{frame_id}.png"
        img  = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_velodyne(self, frame_id: str) -> np.ndarray:
        path = self.vel_dir / f"{frame_id}.bin"
        return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)

    def _load_seg_mask(self, frame_id: str, H: int, W: int) -> np.ndarray:
        """Load KITTI Road drivable area mask (um_ prefix = urban marked)."""
        # KITTI Road files use frame_id with um_ / umm_ / uu_ prefix
        candidates = [
            self.seg_dir / f"um_lane_{frame_id}.png",
            self.seg_dir / f"umm_road_{frame_id}.png",
            self.seg_dir / f"uu_road_{frame_id}.png",
            self.seg_dir / f"{frame_id}.png",
        ]
        for p in candidates:
            if p.exists():
                mask = cv2.imread(str(p), cv2.IMREAD_COLOR)
                # KITTI Road: purple pixels (255,0,255) = road
                road = ((mask[:, :, 0] > 128) & (mask[:, :, 2] > 128)).astype(np.uint8)
                return cv2.resize(road, (W, H), interpolation=cv2.INTER_NEAREST)

        return np.zeros((H, W), dtype=np.uint8)

    def _load_source_frames(self, frame_id: str, offsets: list) -> list:
        """Load adjacent frames for self-supervised depth (returns only valid ones)."""
        idx    = int(frame_id)
        imgs   = []
        for off in offsets:
            if off == 0:
                continue
            fid = f"{idx + off:06d}"
            p   = self.img_dir / f"{fid}.png"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                # Mirror-pad if frame doesn't exist (sequence boundary)
                imgs.append(self._load_image(frame_id).copy())
        return imgs

    def _build_targets(
        self,
        boxes_3d:  np.ndarray,
        class_ids: np.ndarray,
        calib:     KITTICalib,
        H: int, W: int,
    ) -> dict:
        """Render all detection GT tensors from 3D boxes."""
        bev_h = self.bev_cfg["bev_h"]
        bev_w = self.bev_cfg["bev_w"]

        if len(boxes_3d) > 0:
            boxes_bev = box3d_to_bev(
                boxes_3d, self.bev_cfg["pc_range"], bev_h, bev_w
            )
            heatmap, offset, wh = render_heatmap(
                boxes_bev, class_ids, self.num_classes, bev_h, bev_w
            )
        else:
            heatmap = np.zeros((self.num_classes, bev_h, bev_w), dtype=np.float32)
            offset  = np.zeros((2, bev_h, bev_w), dtype=np.float32)
            wh      = np.zeros((2, bev_h, bev_w), dtype=np.float32)

        return {
            "heatmap":    heatmap,       # (C, bev_h, bev_w)
            "offset":     offset,        # (2, bev_h, bev_w)
            "wh":         wh,            # (2, bev_h, bev_w)
            "boxes_3d":   boxes_3d,      # (N, 7) raw for evaluation
            "class_ids":  class_ids,     # (N,)
            # Lane / seg targets are populated after image-space transforms
            "lane_coeffs": np.zeros((6, 4), dtype=np.float32),   # placeholder
            "lane_conf":   np.zeros(6,       dtype=np.float32),
            "seg_mask":    np.zeros((1, H, W), dtype=np.float32),  # filled below
        }

    def __getitem__(self, idx: int) -> dict:
        frame_id = self.frame_ids[idx]

        # ── Raw loads ─────────────────────────────────────────────────────
        image    = self._load_image(frame_id)
        H, W     = image.shape[:2]
        calib    = KITTICalib(str(self.cal_dir / f"{frame_id}.txt"))
        points   = self._load_velodyne(frame_id)
        seg_mask = self._load_seg_mask(frame_id, H, W)
        boxes_3d, class_ids = parse_labels(
            str(self.lbl_dir / f"{frame_id}.txt"), self.class_map
        )

        # LiDAR → image-space sparse depth (used for eval, not training loss)
        depth_map = build_depth_map_from_lidar(
            points, calib.P2, calib.R0, calib.Tr_hom, H, W
        )

        source_imgs = self._load_source_frames(frame_id, self.frame_offsets)

        targets = self._build_targets(boxes_3d, class_ids, calib, H, W)

        # ── Build sample dict for transforms ─────────────────────────────
        sample = {
            "image":       image,
            "depth":       depth_map,
            "seg_mask":    seg_mask,
            "source_imgs": source_imgs,
            "boxes_3d":    boxes_3d,
            "class_ids":   class_ids,
            "calib":       calib.as_dict(),
        }

        sample = self.transforms(sample)

        # Pull seg_mask into targets after transform (shape may have changed)
        if isinstance(sample["seg_mask"], np.ndarray):
            targets["seg_mask"] = sample["seg_mask"][np.newaxis].astype(np.float32)
        else:
            import torch
            targets["seg_mask"] = sample["seg_mask"].unsqueeze(0).float()

        targets["target_img"]  = sample["image"]
        targets["source_imgs"] = sample.get("source_imgs", [])

        return {
            "image":       sample["image"],
            "depth":       sample["depth"],
            "source_imgs": sample.get("source_imgs", []),
            "targets":     targets,
            "calib":       sample["calib"],
            "meta": {
                "frame_id": frame_id,
                "split":    self.split,
                "img_path": str(self.img_dir / f"{frame_id}.png"),
            },
        }

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """
        Custom collate: stack tensors, keep variable-length boxes as lists.
        Needed because each sample may have different numbers of GT boxes.
        """
        import torch

        images       = torch.stack([b["image"] for b in batch])
        depths       = torch.stack([b["depth"] for b in batch])

        # Targets: stack fixed-size tensors, keep variable-size as list
        targets = {}
        for key in ["heatmap", "offset", "wh", "seg_mask"]:
            if key in batch[0]["targets"]:
                targets[key] = torch.stack([b["targets"][key] for b in batch])

        targets["boxes_3d"]  = [b["targets"]["boxes_3d"]  for b in batch]
        targets["class_ids"] = [b["targets"]["class_ids"] for b in batch]
        targets["target_img"]  = images
        targets["source_imgs"] = [b["targets"]["source_imgs"] for b in batch]

        return {
            "image":   images,
            "depth":   depths,
            "targets": targets,
            "calib":   [b["calib"] for b in batch],
            "meta":    [b["meta"]  for b in batch],
        }