"""
Dataset Utilities
=================
Shared helpers used by both KITTI and nuScenes loaders:
  - Gaussian heatmap rendering (CenterPoint-style GT)
  - BEV coordinate transforms
  - Sparse depth map loading + densification
  - Camera projection utilities
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional


# ── Gaussian Heatmap Rendering ─────────────────────────────────────────────

def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.7) -> float:
    """
    Compute Gaussian radius from object size such that the IoU of the
    Gaussian peak with the GT box is >= min_overlap.
    From CenterNet (Zhou et al. 2019).

    Args:
        det_size    : (height, width) of the object in BEV pixels.
        min_overlap : minimum IoU between Gaussian and GT box.
    """
    h, w = det_size
    a1 = 1
    b1 = (h + w)
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)

    return max(0, int(min(r1, r2, r3)))


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
    """
    Draw a 2D Gaussian blob on heatmap at center with given radius.
    Uses element-wise maximum so overlapping objects don't cancel out.

    Args:
        heatmap : (H, W) float32 array, values in [0, 1].
        center  : (cx, cy) in pixel coords.
        radius  : Gaussian radius in pixels.
    Returns:
        heatmap with Gaussian drawn in-place.
    """
    diameter = 2 * radius + 1
    sigma    = diameter / 6.0

    x = np.arange(0, diameter, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = diameter // 2
    gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    cx, cy = int(center[0]), int(center[1])
    H, W   = heatmap.shape

    # Clip Gaussian to heatmap boundaries
    left,   right  = min(cx, radius),       min(W - cx, radius + 1)
    top,    bottom = min(cy, radius),        min(H - cy, radius + 1)
    g_left, g_right  = radius - left,       radius + right
    g_top,  g_bottom = radius - top,        radius + bottom

    heatmap[cy - top : cy + bottom, cx - left : cx + right] = np.maximum(
        heatmap[cy - top : cy + bottom, cx - left : cx + right],
        gaussian[g_top:g_bottom, g_left:g_right],
    )
    return heatmap


def render_heatmap(
    boxes_bev: np.ndarray,
    class_ids: np.ndarray,
    num_classes: int,
    bev_h: int,
    bev_w: int,
    min_overlap: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render multi-class BEV heatmap + offset + wh regression targets.

    Args:
        boxes_bev   : (N, 4) [cx, cy, w, h] in BEV pixel coords.
        class_ids   : (N,) integer class indices.
        num_classes : total number of classes.
        bev_h/w     : BEV grid dimensions.
        min_overlap : Gaussian radius tuning.

    Returns:
        heatmap : (num_classes, bev_h, bev_w) float32
        offset  : (2, bev_h, bev_w) float32 — sub-pixel cx, cy residual
        wh      : (2, bev_h, bev_w) float32 — log(w), log(h)
    """
    heatmap = np.zeros((num_classes, bev_h, bev_w), dtype=np.float32)
    offset  = np.zeros((2, bev_h, bev_w), dtype=np.float32)
    wh      = np.zeros((2, bev_h, bev_w), dtype=np.float32)

    for i, (box, cls) in enumerate(zip(boxes_bev, class_ids)):
        cx, cy, bw, bh = box
        radius = gaussian_radius((bh, bw), min_overlap)
        radius = max(1, int(radius))

        # Integer peak location
        icx, icy = int(cx), int(cy)
        if not (0 <= icx < bev_w and 0 <= icy < bev_h):
            continue

        draw_gaussian(heatmap[cls], (icx, icy), radius)

        # Sub-pixel offset from integer peak
        offset[0, icy, icx] = cx - icx
        offset[1, icy, icx] = cy - icy

        # Log-space dimensions (more Gaussian-like distribution)
        wh[0, icy, icx] = np.log(max(bw, 1e-4))
        wh[1, icy, icx] = np.log(max(bh, 1e-4))

    return heatmap, offset, wh


# ── BEV Coordinate Transforms ──────────────────────────────────────────────

def lidar_to_bev_coords(
    points_xyz: np.ndarray,
    pc_range: List[float],
    bev_h: int,
    bev_w: int,
) -> np.ndarray:
    """
    Map 3D LiDAR points to BEV pixel coordinates.

    Args:
        points_xyz : (N, 3) float32 — x (forward), y (left), z (up)
        pc_range   : [x_min, y_min, z_min, x_max, y_max, z_max]
        bev_h/w    : output BEV grid size

    Returns:
        bev_coords : (N, 2) int32 — (col, row) pixel indices
    """
    x_min, y_min, _, x_max, y_max, _ = pc_range
    x_res = (x_max - x_min) / bev_w
    y_res = (y_max - y_min) / bev_h

    col = ((points_xyz[:, 0] - x_min) / x_res).astype(np.int32)
    row = ((points_xyz[:, 1] - y_min) / y_res).astype(np.int32)

    valid = (col >= 0) & (col < bev_w) & (row >= 0) & (row < bev_h)
    return np.stack([col, row], axis=1), valid


def box3d_to_bev(
    boxes_3d: np.ndarray,
    pc_range: List[float],
    bev_h: int,
    bev_w: int,
) -> np.ndarray:
    """
    Convert 3D boxes [x,y,z,l,w,h,yaw] in LiDAR frame to BEV pixel coords [cx,cy,bw,bh].

    Args:
        boxes_3d : (N, 7) — x, y, z, length, width, height, yaw
        pc_range : [x_min, y_min, z_min, x_max, y_max, z_max]
        bev_h/w  : BEV grid size

    Returns:
        boxes_bev : (N, 4) — cx, cy, bev_w, bev_h in pixel units
    """
    x_min, y_min, _, x_max, y_max, _ = pc_range
    x_res = bev_w / (x_max - x_min)
    y_res = bev_h / (y_max - y_min)

    cx = (boxes_3d[:, 0] - x_min) * x_res
    cy = (boxes_3d[:, 1] - y_min) * y_res
    bw = boxes_3d[:, 3] * x_res    # length → BEV width
    bh = boxes_3d[:, 4] * y_res    # width  → BEV height

    return np.stack([cx, cy, bw, bh], axis=1)


# ── Depth Map Utilities ────────────────────────────────────────────────────

def load_kitti_depth(path: str) -> np.ndarray:
    """
    Load a KITTI depth PNG (16-bit, depth = px / 256.0 metres).
    Returns float32 array with 0.0 for missing values.
    """
    depth_png = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if depth_png is None:
        raise FileNotFoundError(f"Depth file not found: {path}")
    return depth_png.astype(np.float32) / 256.0


def sparse_depth_to_dense(
    sparse: np.ndarray,
    method: str = "nearest",
    max_hole_fill: int = 100,
) -> np.ndarray:
    """
    Fill holes in a sparse depth map (e.g. LiDAR projected onto image).

    Args:
        sparse        : (H, W) float32, 0.0 = missing.
        method        : 'nearest' (fast) or 'ip_basic' (structured fill).
        max_hole_fill : maximum hole diameter to fill (pixels).

    Returns:
        dense : (H, W) float32 — filled depth map.
    """
    if method == "nearest":
        mask   = (sparse == 0).astype(np.uint8)
        filled = cv2.inpaint(
            sparse.astype(np.float32), mask, max_hole_fill, cv2.INPAINT_NS
        )
        return filled

    elif method == "ip_basic":
        # Fast IP-Basic: morphological fill (Ku et al. 2018)
        depth = sparse.copy()
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(max_hole_fill // 3):
            mask = (depth == 0).astype(np.uint8)
            if mask.sum() == 0:
                break
            depth_dilated = cv2.dilate(depth, kernel)
            depth[mask.astype(bool)] = depth_dilated[mask.astype(bool)]
        return depth

    else:
        raise ValueError(f"Unknown densification method: {method}")


# ── Camera Projection ──────────────────────────────────────────────────────

def project_lidar_to_image(
    points: np.ndarray,
    P: np.ndarray,
    R0: np.ndarray,
    Tr: np.ndarray,
    img_h: int,
    img_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project LiDAR points onto the image plane (KITTI convention).

    Args:
        points : (N, 3) or (N, 4) LiDAR points in velodyne frame.
        P      : (3, 4) camera projection matrix.
        R0     : (3, 3) rectification matrix.
        Tr     : (3, 4) velodyne→camera rigid transform.
        img_h/w: image dimensions for bounds checking.

    Returns:
        uv     : (M, 2) int32 pixel coordinates of valid projected points.
        depths : (M,)   float32 depth values of valid projected points.
    """
    pts = points[:, :3]
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_hom = np.hstack([pts, ones])          # (N, 4)

    # Velodyne → camera rect coords
    R0_ext = np.eye(4, dtype=np.float32)
    R0_ext[:3, :3] = R0
    cam_pts = (R0_ext @ Tr @ pts_hom.T)       # (4, N)

    # Keep only points in front of camera
    valid_depth = cam_pts[2, :] > 0
    cam_pts = cam_pts[:, valid_depth]

    # Project to image
    img_pts = P @ cam_pts                      # (3, M)
    img_pts /= img_pts[2:3, :]                # normalise

    u = img_pts[0, :].astype(np.int32)
    v = img_pts[1, :].astype(np.int32)
    d = cam_pts[2, :]

    in_bounds = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return np.stack([u[in_bounds], v[in_bounds]], axis=1), d[in_bounds]


def build_depth_map_from_lidar(
    points: np.ndarray,
    P: np.ndarray,
    R0: np.ndarray,
    Tr: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Project LiDAR points to build a sparse depth map (H, W)."""
    uv, depths = project_lidar_to_image(points, P, R0, Tr, img_h, img_w)
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    if uv.shape[0] > 0:
        depth_map[uv[:, 1], uv[:, 0]] = depths
    return depth_map