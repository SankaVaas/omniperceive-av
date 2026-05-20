"""
Joint Augmentation Pipeline
============================
All transforms operate on a unified sample dict so that every modality
(RGB, depth, boxes, lanes, segmentation mask) is augmented *consistently*.

Transforms are composable via `Compose` and configured entirely from YAML.

Sample dict keys (all optional — transforms check before applying):
    image       : (H, W, 3)  uint8  RGB
    depth       : (H, W)     float32
    seg_mask    : (H, W)     uint8   binary / multi-class
    boxes_2d    : (N, 4)     float32 [x1,y1,x2,y2]
    boxes_3d    : (N, 7)     float32 [x,y,z,l,w,h,yaw] in LiDAR frame
    lane_pts    : list of (K,2) float32 — pixel-space lane points
    calib       : dict with P2, R0, Tr_velo_cam (KITTI) or camera matrices (nuScenes)
    source_imgs : list of (H, W, 3) adjacent frames for depth self-supervision
"""

import cv2
import numpy as np
from typing import Dict, List, Callable, Optional, Tuple


class Compose:
    """Chain multiple transforms together."""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        lines = [self.__class__.__name__ + "("]
        for t in self.transforms:
            lines.append(f"    {t},")
        lines.append(")")
        return "\n".join(lines)


class Resize:
    """
    Resize image (and all spatial modalities) to target (H, W).
    Adjusts camera intrinsics (P2) proportionally.
    """
    def __init__(self, height: int, width: int):
        self.h = height
        self.w = width

    def __call__(self, sample: dict) -> dict:
        oh, ow = sample["image"].shape[:2]
        sy, sx = self.h / oh, self.w / ow

        sample["image"] = cv2.resize(
            sample["image"], (self.w, self.h), interpolation=cv2.INTER_LINEAR
        )

        if "depth" in sample:
            # Use nearest to avoid interpolating valid/invalid depth boundary
            sample["depth"] = cv2.resize(
                sample["depth"], (self.w, self.h), interpolation=cv2.INTER_NEAREST
            )

        if "seg_mask" in sample:
            sample["seg_mask"] = cv2.resize(
                sample["seg_mask"], (self.w, self.h), interpolation=cv2.INTER_NEAREST
            )

        if "boxes_2d" in sample and len(sample["boxes_2d"]) > 0:
            b = sample["boxes_2d"].copy()
            b[:, [0, 2]] *= sx
            b[:, [1, 3]] *= sy
            sample["boxes_2d"] = b

        if "lane_pts" in sample:
            scaled = []
            for lane in sample["lane_pts"]:
                l = lane.copy().astype(np.float32)
                l[:, 0] *= sx
                l[:, 1] *= sy
                scaled.append(l)
            sample["lane_pts"] = scaled

        # Scale intrinsic matrix
        if "calib" in sample and "P2" in sample["calib"]:
            P = sample["calib"]["P2"].copy()
            P[0, :] *= sx   # fx, cx
            P[1, :] *= sy   # fy, cy
            sample["calib"]["P2"] = P

        if "source_imgs" in sample:
            sample["source_imgs"] = [
                cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
                for img in sample["source_imgs"]
            ]

        return sample


class RandomCrop:
    """
    Random crop to target (H, W). Applied consistently across all modalities.
    Biased toward bottom of image to preserve road area.
    """
    def __init__(self, height: int, width: int, bias_bottom: bool = True):
        self.h = height
        self.w = width
        self.bias_bottom = bias_bottom

    def __call__(self, sample: dict) -> dict:
        oh, ow = sample["image"].shape[:2]
        assert oh >= self.h and ow >= self.w, \
            f"Image {oh}×{ow} smaller than crop {self.h}×{self.w}"

        if self.bias_bottom:
            top = oh - self.h           # always crop from bottom
        else:
            top = np.random.randint(0, oh - self.h + 1)
        left = np.random.randint(0, ow - self.w + 1)

        def _crop_img(img):
            return img[top:top + self.h, left:left + self.w]

        sample["image"] = _crop_img(sample["image"])

        if "depth" in sample:
            sample["depth"] = _crop_img(sample["depth"])

        if "seg_mask" in sample:
            sample["seg_mask"] = _crop_img(sample["seg_mask"])

        if "boxes_2d" in sample and len(sample["boxes_2d"]) > 0:
            b = sample["boxes_2d"].copy()
            b[:, [0, 2]] -= left
            b[:, [1, 3]] -= top
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, self.w)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, self.h)
            # Remove boxes that became too small after crop
            keep = ((b[:, 2] - b[:, 0]) > 2) & ((b[:, 3] - b[:, 1]) > 2)
            sample["boxes_2d"]  = b[keep]
            if "boxes_3d" in sample:
                sample["boxes_3d"] = sample["boxes_3d"][keep]
            if "class_ids" in sample:
                sample["class_ids"] = sample["class_ids"][keep]

        if "lane_pts" in sample:
            cropped = []
            for lane in sample["lane_pts"]:
                l = lane.copy().astype(np.float32)
                l[:, 0] -= left
                l[:, 1] -= top
                in_crop = ((l[:, 0] >= 0) & (l[:, 0] < self.w) &
                           (l[:, 1] >= 0) & (l[:, 1] < self.h))
                if in_crop.sum() >= 2:
                    cropped.append(l[in_crop])
            sample["lane_pts"] = cropped

        # Adjust intrinsics for crop offset
        if "calib" in sample and "P2" in sample["calib"]:
            P = sample["calib"]["P2"].copy()
            P[0, 2] -= left   # cx
            P[1, 2] -= top    # cy
            sample["calib"]["P2"] = P

        if "source_imgs" in sample:
            sample["source_imgs"] = [_crop_img(img) for img in sample["source_imgs"]]

        return sample


class RandomHorizontalFlip:
    """
    Flip image left-right with probability p.
    Flips 3D box yaw angles and lane points consistently.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if np.random.random() >= self.p:
            return sample

        W = sample["image"].shape[1]

        sample["image"] = np.fliplr(sample["image"]).copy()

        if "depth" in sample:
            sample["depth"] = np.fliplr(sample["depth"]).copy()

        if "seg_mask" in sample:
            sample["seg_mask"] = np.fliplr(sample["seg_mask"]).copy()

        if "boxes_2d" in sample and len(sample["boxes_2d"]) > 0:
            b = sample["boxes_2d"].copy()
            b[:, [0, 2]] = W - b[:, [2, 0]]
            sample["boxes_2d"] = b

        if "boxes_3d" in sample and len(sample["boxes_3d"]) > 0:
            b3 = sample["boxes_3d"].copy()
            b3[:, 1] = -b3[:, 1]           # y_center flipped
            b3[:, 6] = -b3[:, 6] + np.pi  # yaw angle flipped
            sample["boxes_3d"] = b3

        if "lane_pts" in sample:
            flipped = []
            for lane in sample["lane_pts"]:
                l = lane.copy()
                l[:, 0] = W - l[:, 0]
                flipped.append(l)
            sample["lane_pts"] = flipped

        if "calib" in sample and "P2" in sample["calib"]:
            P = sample["calib"]["P2"].copy()
            P[0, 2] = W - P[0, 2]         # cx flipped
            sample["calib"]["P2"] = P

        if "source_imgs" in sample:
            sample["source_imgs"] = [
                np.fliplr(img).copy() for img in sample["source_imgs"]
            ]

        return sample


class PhotometricDistortion:
    """
    Random brightness / contrast / saturation / hue jitter.
    Applied identically to all frames (target + source) for depth consistency.
    Only affects RGB — depth, masks, boxes unchanged.
    """
    def __init__(
        self,
        brightness: float = 0.2,
        contrast:   float = 0.2,
        saturation: float = 0.2,
        hue:        float = 0.05,
        p:          float = 0.5,
    ):
        self.b  = brightness
        self.c  = contrast
        self.s  = saturation
        self.h  = hue
        self.p  = p

    def _jitter(self, img: np.ndarray, b: float, c: float, s: float, h: float) -> np.ndarray:
        """Apply fixed jitter params to one image."""
        img = img.astype(np.float32) / 255.0
        img = np.clip(img * (1 + c) + b, 0, 1)  # brightness + contrast

        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + s), 0, 255)   # saturation
        hsv[:, :, 0] = (hsv[:, :, 0] + h * 180) % 180             # hue
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img

    def __call__(self, sample: dict) -> dict:
        if np.random.random() >= self.p:
            return sample

        b = np.random.uniform(-self.b, self.b)
        c = np.random.uniform(-self.c, self.c)
        s = np.random.uniform(-self.s, self.s)
        h = np.random.uniform(-self.h, self.h)

        sample["image"] = self._jitter(sample["image"], b, c, s, h)

        if "source_imgs" in sample:
            sample["source_imgs"] = [
                self._jitter(img, b, c, s, h) for img in sample["source_imgs"]
            ]

        return sample


class Normalize:
    """Normalize image to float32 with ImageNet mean/std. Converts HWC→CHW."""
    def __init__(
        self,
        mean: List[float] = [0.485, 0.456, 0.406],
        std:  List[float] = [0.229, 0.224, 0.225],
    ):
        self.mean = np.array(mean, dtype=np.float32)
        self.std  = np.array(std,  dtype=np.float32)

    def __call__(self, sample: dict) -> dict:
        img = sample["image"].astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        sample["image"] = img.transpose(2, 0, 1)   # HWC → CHW

        if "source_imgs" in sample:
            normed = []
            for src in sample["source_imgs"]:
                s = src.astype(np.float32) / 255.0
                s = (s - self.mean) / self.std
                normed.append(s.transpose(2, 0, 1))
            sample["source_imgs"] = normed

        return sample


class ToTensor:
    """Convert numpy arrays in sample dict to torch tensors."""
    def __call__(self, sample: dict) -> dict:
        import torch
        for key in ["image", "depth", "seg_mask"]:
            if key in sample and isinstance(sample[key], np.ndarray):
                sample[key] = torch.from_numpy(sample[key].copy())

        if "source_imgs" in sample:
            sample["source_imgs"] = [
                torch.from_numpy(s.copy()) for s in sample["source_imgs"]
            ]

        for key in ["boxes_2d", "boxes_3d", "class_ids",
                    "heatmap", "offset", "wh"]:
            if key in sample and isinstance(sample[key], np.ndarray):
                sample[key] = torch.from_numpy(sample[key].copy())

        return sample


def build_transforms(cfg: dict, split: str = "train") -> Compose:
    """
    Build transform pipeline from config dict.

    Args:
        cfg   : training.augmentation section of YAML config.
        split : 'train' or 'val'. Val only gets Resize + Normalize + ToTensor.
    """
    aug = cfg.get("augmentation", {})
    h   = cfg["dataset"]["img_height"]
    w   = cfg["dataset"]["img_width"]

    if split == "val":
        return Compose([
            Resize(h, w),
            Normalize(aug.get("normalize", {}).get("mean", [0.485, 0.456, 0.406]),
                      aug.get("normalize", {}).get("std",  [0.229, 0.224, 0.225])),
            ToTensor(),
        ])

    # Training pipeline
    transforms = [Resize(h + 32, w + 64)]  # slight oversize before crop

    crop = aug.get("random_crop")
    if crop:
        transforms.append(RandomCrop(crop[0], crop[1], bias_bottom=True))
    else:
        transforms.append(Resize(h, w))

    if aug.get("random_flip_horizontal", 0) > 0:
        transforms.append(RandomHorizontalFlip(p=aug["random_flip_horizontal"]))

    jitter = aug.get("color_jitter", {})
    if jitter:
        transforms.append(PhotometricDistortion(
            brightness=jitter.get("brightness", 0.2),
            contrast=jitter.get("contrast",     0.2),
            saturation=jitter.get("saturation", 0.2),
            hue=jitter.get("hue",               0.05),
        ))

    norm = aug.get("normalize", {})
    transforms.append(Normalize(
        mean=norm.get("mean", [0.485, 0.456, 0.406]),
        std=norm.get("std",   [0.229, 0.224, 0.225]),
    ))
    transforms.append(ToTensor())

    return Compose(transforms)