"""
nuScenes Multi-Task Dataset
============================
Loads all six surround cameras + LiDAR for multi-task training.

Requires: nuscenes-devkit  (pip install nuscenes-devkit)

Data layout:
  data/nuscenes/
    v1.0-trainval/   ← annotation JSON files
    samples/         ← keyframe sensor data
    sweeps/          ← intermediate LiDAR sweeps
    maps/            ← HD map tiles

Per sample output:
    images       : (6, 3, H, W)  — all 6 cameras, ordered per self.cameras
    depth_maps   : (6, 1, H, W)  — LiDAR projected per camera (eval only)
    seg_masks    : (6, 2, H, W)  — HD map rasterised: drivable + lane divider
    source_imgs  : (6, 2, 3, H, W) — prev/next keyframe for depth self-supervision
    targets      : heatmap/offset/wh on unified BEV, lane/seg labels
    calib        : list of 6 dicts (intrinsics + extrinsics per camera)
    meta         : sample_token, scene_name, timestamp
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset

from datasets.transforms import build_transforms
from datasets.utils import render_heatmap, box3d_to_bev

# nuScenes 10-class mapping (standard detection benchmark)
NUSCENES_CLASSES = {
    "car":                    0,
    "truck":                  1,
    "construction_vehicle":   2,
    "bus":                    3,
    "trailer":                4,
    "barrier":                5,
    "motorcycle":             6,
    "bicycle":                7,
    "pedestrian":             8,
    "traffic_cone":           9,
}

# nuScenes camera names in order (front → back, clockwise)
CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class NuScenesCalib:
    """Per-camera calibration: intrinsics (K) + extrinsics (cam2ego, ego2global)."""

    def __init__(self, cam_data: dict, nusc):
        cs_record  = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        ego_record = nusc.get("ego_pose",           cam_data["ego_pose_token"])

        self.K = np.array(cs_record["camera_intrinsic"], dtype=np.float32)  # (3,3)

        # Rotation + translation: sensor → ego vehicle frame
        from pyquaternion import Quaternion
        self.cam2ego_rot = Quaternion(cs_record["rotation"]).rotation_matrix.astype(np.float32)
        self.cam2ego_t   = np.array(cs_record["translation"], dtype=np.float32)

        # Ego → global (world) frame at this timestamp
        self.ego2global_rot = Quaternion(ego_record["rotation"]).rotation_matrix.astype(np.float32)
        self.ego2global_t   = np.array(ego_record["translation"], dtype=np.float32)

    def as_dict(self) -> dict:
        return {
            "K":              self.K,
            "cam2ego_rot":    self.cam2ego_rot,
            "cam2ego_t":      self.cam2ego_t,
            "ego2global_rot": self.ego2global_rot,
            "ego2global_t":   self.ego2global_t,
        }


class NuScenesMultiTaskDataset(Dataset):
    """
    nuScenes multi-camera, multi-task dataset.

    Args:
        cfg   : full YAML config dict (dataset + model + training sections).
        split : 'train' or 'val'.

    Colab/Kaggle note:
        Set cfg.dataset.version = 'v1.0-mini' for quick iteration.
        Mini has 10 scenes (~400 keyframes) and downloads in minutes.
    """

    def __init__(self, cfg: dict, split: str = "train"):
        self.cfg    = cfg
        self.split  = split
        self.root   = Path(cfg["dataset"]["root"])
        self.version = cfg["dataset"].get("version", "v1.0-trainval")
        self.img_h  = cfg["dataset"]["img_height"]
        self.img_w  = cfg["dataset"]["img_width"]
        self.cameras = cfg["dataset"].get("cameras", CAMERA_NAMES)
        self.num_cameras = len(self.cameras)
        self.bev_cfg = {
            "pc_range": cfg["model"]["bev_neck"]["pc_range"],
            "bev_h":    cfg["model"]["bev_neck"]["bev_h"],
            "bev_w":    cfg["model"]["bev_neck"]["bev_w"],
        }
        self.num_classes    = len(NUSCENES_CLASSES)
        self.lidar_sweeps   = cfg["dataset"].get("lidar_sweeps", 10)
        self.use_map_gt     = cfg["dataset"].get("use_map_gt", True)

        # ── Initialise nuScenes devkit ────────────────────────────────────
        try:
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError:
            raise ImportError(
                "nuScenes devkit not installed.\n"
                "Run: pip install nuscenes-devkit"
            )

        print(f"[nuScenes] Loading {self.version} from {self.root} …")
        self.nusc = NuScenes(version=self.version, dataroot=str(self.root), verbose=False)

        splits      = create_splits_scenes()
        split_key   = "train" if split == "train" else "val"
        scene_names = set(splits[split_key])

        # Collect all keyframe sample tokens for the split
        self.sample_tokens = []
        for scene in self.nusc.scene:
            if scene["name"] not in scene_names:
                continue
            sample_token = scene["first_sample_token"]
            while sample_token:
                self.sample_tokens.append(sample_token)
                sample = self.nusc.get("sample", sample_token)
                sample_token = sample["next"]

        print(f"[nuScenes] {split} split: {len(self.sample_tokens)} keyframes, "
              f"{self.num_cameras} cameras each.")

        self.transforms = build_transforms(cfg["training"], split)

    def __len__(self):
        return len(self.sample_tokens)

    def _load_camera(self, cam_token: str) -> Tuple[np.ndarray, NuScenesCalib]:
        """Load one camera image and its calibration."""
        cam_data = self.nusc.get("sample_data", cam_token)
        img_path = self.root / cam_data["filename"]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        calib = NuScenesCalib(cam_data, self.nusc)
        return img, calib

    def _load_lidar_points(self, sample: dict) -> np.ndarray:
        """
        Load LiDAR point cloud, optionally accumulating N sweeps for denser coverage.
        Returns (N, 4) float32 in ego vehicle frame at keyframe timestamp.
        """
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion

        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data  = self.nusc.get("sample_data", lidar_token)

        pc = LidarPointCloud.from_file(str(self.root / lidar_data["filename"]))

        # Aggregate past sweeps → denser point cloud
        num_sweeps = self.lidar_sweeps
        current = lidar_data
        for _ in range(num_sweeps - 1):
            if current["prev"] == "":
                break
            current  = self.nusc.get("sample_data", current["prev"])
            pc_sweep = LidarPointCloud.from_file(str(self.root / current["filename"]))

            # Transform sweep into keyframe ego frame
            cs       = self.nusc.get("calibrated_sensor", current["calibrated_sensor_token"])
            ego_pose = self.nusc.get("ego_pose",          current["ego_pose_token"])
            kf_cs    = self.nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
            kf_ego   = self.nusc.get("ego_pose",          lidar_data["ego_pose_token"])

            T = (
                np.linalg.inv(transform_matrix(kf_ego["translation"], Quaternion(kf_ego["rotation"]))) @
                transform_matrix(ego_pose["translation"], Quaternion(ego_pose["rotation"])) @
                transform_matrix(cs["translation"], Quaternion(cs["rotation"])) @
                np.linalg.inv(transform_matrix(kf_cs["translation"], Quaternion(kf_cs["rotation"])))
            )
            pc_sweep.transform(T)
            pc.points = np.hstack([pc.points, pc_sweep.points])

        return pc.points[:4, :].T.astype(np.float32)   # (N, 4) xyzr in ego frame

    def _project_lidar_to_camera(
        self,
        points_ego: np.ndarray,
        calib: NuScenesCalib,
        H: int, W: int,
    ) -> np.ndarray:
        """Project LiDAR (ego frame) → image pixels → sparse depth map."""
        # Ego → camera frame
        pts = points_ego[:, :3]                         # (N, 3)
        pts_cam = (calib.cam2ego_rot.T @ (pts - calib.cam2ego_t).T).T  # (N, 3)

        # Keep only in front of camera
        valid = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[valid]

        # Project
        uvd = (calib.K @ pts_cam.T).T                  # (M, 3)
        uvd /= uvd[:, 2:3]

        u = uvd[:, 0].astype(np.int32)
        v = uvd[:, 1].astype(np.int32)
        d = pts_cam[:, 2]

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        depth_map = np.zeros((H, W), dtype=np.float32)
        if in_bounds.any():
            # If multiple points map to same pixel, keep closest
            idx = np.argsort(-d[in_bounds])             # sort far→near (near overwrites)
            u_v, v_v, d_v = u[in_bounds][idx], v[in_bounds][idx], d[in_bounds][idx]
            depth_map[v_v, u_v] = d_v

        return depth_map

    def _rasterise_map(
        self,
        sample_token: str,
        calib: NuScenesCalib,
        H: int, W: int,
        layers: List[str] = ["drivable_area", "lane_divider"],
    ) -> np.ndarray:
        """
        Rasterise nuScenes HD map layers into image space.
        Returns (num_layers, H, W) uint8 binary masks.
        """
        try:
            from nuscenes.map_expansion.map_api import NuScenesMap
        except ImportError:
            # Map expansion not available — return zeros
            return np.zeros((len(layers), H, W), dtype=np.uint8)

        sample = self.nusc.get("sample", sample_token)
        scene  = self.nusc.get("scene", sample["scene_token"])
        log    = self.nusc.get("log",   scene["log_token"])
        map_name = log["location"]

        nusc_map = NuScenesMap(dataroot=str(self.root), map_name=map_name)

        # Ego pose at this keyframe (use LIDAR_TOP for stable ego frame)
        lidar_data  = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_record  = self.nusc.get("ego_pose",    lidar_data["ego_pose_token"])
        from pyquaternion import Quaternion
        ego_t = np.array(ego_record["translation"][:2])   # (x, y) in global

        masks = []
        patch_size = 50.0  # metres around ego to sample
        for layer in layers:
            try:
                mask_bev = nusc_map.get_map_mask(
                    patch_box=(ego_t[0], ego_t[1], patch_size * 2, patch_size * 2),
                    patch_angle=0.0,
                    layer_names=[layer],
                    canvas_size=(H, W),
                )[0]
                masks.append(mask_bev.astype(np.uint8))
            except Exception:
                masks.append(np.zeros((H, W), dtype=np.uint8))

        return np.stack(masks, axis=0)  # (num_layers, H, W)

    def _get_boxes_in_ego(self, sample: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve all annotated 3D boxes in ego vehicle frame.
        Returns boxes_3d (N,7) [x,y,z,l,w,h,yaw] and class_ids (N,).
        """
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion

        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data  = self.nusc.get("sample_data", lidar_token)
        ego_pose    = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])

        T_global2ego = np.linalg.inv(
            transform_matrix(ego_pose["translation"], Quaternion(ego_pose["rotation"]))
        )

        boxes, cls_ids = [], []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)

            cat = ann["category_name"].split(".")[0]  # e.g. "vehicle" → check below
            # Map nuScenes full category to 10-class benchmark
            mapped = None
            for cls_name in NUSCENES_CLASSES:
                if cls_name in ann["category_name"]:
                    mapped = NUSCENES_CLASSES[cls_name]
                    break
            if mapped is None:
                continue

            # Global → ego
            t_global = np.array(ann["translation"] + [1.0])
            t_ego    = T_global2ego @ t_global

            from pyquaternion import Quaternion as Q
            q_global = Q(ann["rotation"])
            # Yaw in ego frame
            ego_q = Q(ego_pose["rotation"]).inverse * q_global
            yaw   = ego_q.yaw_pitch_roll[0]

            l, w, h = ann["size"]   # nuScenes: [width, length, height] → reorder
            boxes.append([t_ego[0], t_ego[1], t_ego[2], l, w, h, yaw])
            cls_ids.append(mapped)

        if not boxes:
            return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.int64)

        return np.array(boxes, dtype=np.float32), np.array(cls_ids, dtype=np.int64)

    def _load_adjacent_keyframe(
        self, sample_token: str, direction: str, cam_name: str
    ) -> Optional[np.ndarray]:
        """Load adjacent keyframe image for self-supervised depth (prev/next)."""
        sample = self.nusc.get("sample", sample_token)
        adj_token = sample[direction]           # 'prev' or 'next'
        if not adj_token:
            return None
        adj_sample = self.nusc.get("sample", adj_token)
        cam_token  = adj_sample["data"].get(cam_name)
        if not cam_token:
            return None
        cam_data = self.nusc.get("sample_data", cam_token)
        img = cv2.imread(str(self.root / cam_data["filename"]), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> dict:
        sample_token = self.sample_tokens[idx]
        sample       = self.nusc.get("sample", sample_token)

        # ── Per-camera loads ──────────────────────────────────────────────
        all_images, all_calibs, all_depths, all_segs, all_sources = [], [], [], [], []

        lidar_pts = self._load_lidar_points(sample)

        for cam_name in self.cameras:
            cam_token = sample["data"][cam_name]
            img, calib = self._load_camera(cam_token)
            H_raw, W_raw = img.shape[:2]

            # Sparse depth from LiDAR projection
            depth_map = self._project_lidar_to_camera(lidar_pts, calib, H_raw, W_raw)

            # HD map segmentation
            if self.use_map_gt:
                seg = self._rasterise_map(sample_token, calib, H_raw, W_raw)
            else:
                seg = np.zeros((2, H_raw, W_raw), dtype=np.uint8)

            # Adjacent keyframes for self-supervised depth
            src_imgs = []
            for direction in ["prev", "next"]:
                adj = self._load_adjacent_keyframe(sample_token, direction, cam_name)
                src_imgs.append(adj if adj is not None else img.copy())

            # Apply transforms per camera
            cam_sample = {
                "image":       img,
                "depth":       depth_map,
                "seg_mask":    seg.transpose(1, 2, 0),   # CHW→HWC for transforms
                "source_imgs": src_imgs,
                "calib":       {"P2": calib.K},          # transforms expect P2 key
            }
            cam_sample = self.transforms(cam_sample)

            all_images.append(cam_sample["image"])
            all_calibs.append(calib.as_dict())
            all_depths.append(cam_sample["depth"])
            all_segs.append(cam_sample["seg_mask"])
            all_sources.append(cam_sample.get("source_imgs", []))

        # ── BEV detection targets (all cameras share one BEV grid) ────────
        boxes_3d, class_ids = self._get_boxes_in_ego(sample)

        if len(boxes_3d) > 0:
            boxes_bev = box3d_to_bev(
                boxes_3d, self.bev_cfg["pc_range"],
                self.bev_cfg["bev_h"], self.bev_cfg["bev_w"]
            )
            heatmap, offset, wh = render_heatmap(
                boxes_bev, class_ids, self.num_classes,
                self.bev_cfg["bev_h"], self.bev_cfg["bev_w"]
            )
        else:
            bev_h, bev_w = self.bev_cfg["bev_h"], self.bev_cfg["bev_w"]
            heatmap = np.zeros((self.num_classes, bev_h, bev_w), dtype=np.float32)
            offset  = np.zeros((2, bev_h, bev_w), dtype=np.float32)
            wh      = np.zeros((2, bev_h, bev_w), dtype=np.float32)

        import torch, numpy as np_

        targets = {
            "heatmap":    torch.from_numpy(heatmap),
            "offset":     torch.from_numpy(offset),
            "wh":         torch.from_numpy(wh),
            "boxes_3d":   boxes_3d,
            "class_ids":  class_ids,
            "seg_masks":  all_segs,         # list of 6 tensors (2,H,W)
            "target_imgs":  all_images,     # list of 6 tensors (3,H,W) for depth loss
            "source_imgs":  all_sources,    # list of 6 × [prev, next]
            "lane_coeffs": np_.zeros((8, 4), dtype=np_.float32),
            "lane_conf":   np_.zeros(8,       dtype=np_.float32),
        }

        return {
            "images":      torch.stack(all_images),    # (6, 3, H, W)
            "depths":      torch.stack([torch.from_numpy(d).unsqueeze(0) for d in all_depths]),
            "seg_masks":   all_segs,
            "targets":     targets,
            "calibs":      all_calibs,
            "meta": {
                "sample_token": sample_token,
                "scene_name":   self.nusc.get("scene", sample["scene_token"])["name"],
                "timestamp":    sample["timestamp"],
            },
        }

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """Collate multi-camera nuScenes batch."""
        import torch
        images  = torch.stack([b["images"]  for b in batch])    # (B, 6, 3, H, W)
        depths  = torch.stack([b["depths"]  for b in batch])    # (B, 6, 1, H, W)

        targets = {}
        for key in ["heatmap", "offset", "wh"]:
            targets[key] = torch.stack([b["targets"][key] for b in batch])

        targets["boxes_3d"]  = [b["targets"]["boxes_3d"]  for b in batch]
        targets["class_ids"] = [b["targets"]["class_ids"] for b in batch]
        targets["target_imgs"]  = [b["targets"]["target_imgs"]  for b in batch]
        targets["source_imgs"]  = [b["targets"]["source_imgs"]  for b in batch]
        targets["lane_coeffs"] = torch.from_numpy(
            np.stack([b["targets"]["lane_coeffs"] for b in batch])
        )
        targets["lane_conf"]   = torch.from_numpy(
            np.stack([b["targets"]["lane_conf"]   for b in batch])
        )

        return {
            "images":  images,
            "depths":  depths,
            "targets": targets,
            "calibs":  [b["calibs"] for b in batch],
            "meta":    [b["meta"]   for b in batch],
        }