"""
BEV Utilities
==============
Helpers for working in Bird's-Eye-View space:
  - Coordinate conversion  (LiDAR ↔ BEV pixel ↔ ego metres)
  - Rotated box rendering  (for oriented 3D detections in BEV)
  - BEV feature visualisation helpers
  - Voxelisation for point clouds
"""

import numpy as np
import cv2
from typing import List, Tuple


# ── Coordinate conversions ─────────────────────────────────────────────────

def world_to_bev_pixel(
    x_world: np.ndarray,
    y_world: np.ndarray,
    pc_range: List[float],
    bev_h: int,
    bev_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert ego/LiDAR (x, y) metres → BEV pixel coordinates (col, row).

    Convention (KITTI / standard):
      x = forward  → maps to BEV columns  (left = col 0)
      y = left     → maps to BEV rows     (bottom = row 0 in Cartesian,
                                           but row 0 = top in image space)

    Args:
        x_world, y_world : arrays of world coordinates in metres.
        pc_range         : [x_min, y_min, z_min, x_max, y_max, z_max]
        bev_h, bev_w     : BEV grid dimensions.

    Returns:
        col, row : integer pixel coordinates (clipped to grid bounds).
    """
    x_min, y_min, _, x_max, y_max, _ = pc_range
    x_res = bev_w / (x_max - x_min)
    y_res = bev_h / (y_max - y_min)

    col = np.clip(((x_world - x_min) * x_res).astype(int), 0, bev_w - 1)
    row = np.clip(((y_world - y_min) * y_res).astype(int), 0, bev_h - 1)
    return col, row


def bev_pixel_to_world(
    col: np.ndarray,
    row: np.ndarray,
    pc_range: List[float],
    bev_h: int,
    bev_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of world_to_bev_pixel: BEV pixel → ego metres."""
    x_min, y_min, _, x_max, y_max, _ = pc_range
    x_res = (x_max - x_min) / bev_w
    y_res = (y_max - y_min) / bev_h

    x_world = col * x_res + x_min
    y_world = row * y_res + y_min
    return x_world, y_world


# ── Rotated bounding box rendering ────────────────────────────────────────

def get_rotated_box_corners(
    cx: float, cy: float,
    w:  float, h:  float,
    yaw: float,
) -> np.ndarray:
    """
    Compute 4 corner coordinates of a rotated BEV box.

    Args:
        cx, cy : centre in BEV pixels.
        w, h   : box width and height in BEV pixels.
        yaw    : rotation angle in radians (counter-clockwise).

    Returns:
        corners : (4, 2) float array of corner (col, row) coordinates.
    """
    cos_a, sin_a = np.cos(yaw), np.sin(yaw)

    # Half-extents
    hw, hh = w / 2, h / 2

    # Local corners (before rotation)
    local = np.array([
        [ hw,  hh],
        [-hw,  hh],
        [-hw, -hh],
        [ hw, -hh],
    ])  # (4, 2)

    # Rotation matrix
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])

    rotated = local @ R.T   # (4, 2)
    return rotated + np.array([cx, cy])


def draw_rotated_box_bev(
    canvas:  np.ndarray,   # (H, W, 3) uint8 BGR canvas
    cx:      float,
    cy:      float,
    w:       float,
    h:       float,
    yaw:     float,
    color:   Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a single rotated BEV box on canvas. Returns canvas."""
    corners = get_rotated_box_corners(cx, cy, w, h, yaw).astype(np.int32)
    # Draw 4 edges
    for i in range(4):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i + 1) % 4])
        cv2.line(canvas, pt1, pt2, color, thickness)
    # Arrow from centre to front face (between corners 0 and 3) to show heading
    front_mid = ((corners[0] + corners[3]) / 2).astype(int)
    cv2.arrowedLine(canvas, (int(cx), int(cy)), tuple(front_mid),
                    color, thickness, tipLength=0.3)
    return canvas


# ── BEV canvas construction ────────────────────────────────────────────────

CLASS_COLORS_BGR = {
    0: (255, 128,   0),    # Car       — orange
    1: (  0, 255, 255),    # Pedestrian — yellow
    2: (255,   0, 255),    # Cyclist   — magenta
    3: (128, 255, 128),    # Truck
    4: (  0, 128, 255),    # Bus
}


def render_bev_detections(
    bev_h:   int,
    bev_w:   int,
    boxes_bev: np.ndarray,     # (N, 4) [cx, cy, bev_w, bev_h] in pixels
    class_ids: np.ndarray,     # (N,)
    scores:    np.ndarray,     # (N,)
    yaws:      np.ndarray = None,  # (N,) optional rotation angles
    score_thresh: float = 0.3,
) -> np.ndarray:
    """
    Render predicted 3D detections onto a blank BEV canvas.

    Returns:
        canvas : (bev_h, bev_w, 3) uint8 BGR image.
    """
    canvas = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)   # dark background

    for i, (box, cls, score) in enumerate(zip(boxes_bev, class_ids, scores)):
        if score < score_thresh:
            continue
        cx, cy, bw, bh = box
        color = CLASS_COLORS_BGR.get(int(cls), (200, 200, 200))
        yaw   = yaws[i] if yaws is not None else 0.0

        canvas = draw_rotated_box_bev(canvas, cx, cy, bw, bh, yaw,
                                      color=color, thickness=2)
        # Score label
        cv2.putText(canvas, f"{score:.2f}",
                    (int(cx), int(cy) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return canvas


def render_bev_gt(
    bev_h: int,
    bev_w: int,
    boxes_bev: np.ndarray,
    class_ids: np.ndarray,
    yaws: np.ndarray = None,
) -> np.ndarray:
    """Render GT boxes on BEV canvas (white boxes, no scores)."""
    canvas = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)
    for i, (box, cls) in enumerate(zip(boxes_bev, class_ids)):
        cx, cy, bw, bh = box
        yaw = yaws[i] if yaws is not None else 0.0
        color = CLASS_COLORS_BGR.get(int(cls), (255, 255, 255))
        canvas = draw_rotated_box_bev(canvas, cx, cy, bw, bh, yaw,
                                      color=color, thickness=1)
    return canvas


# ── Voxelisation ──────────────────────────────────────────────────────────

def voxelise_point_cloud(
    points:   np.ndarray,     # (N, 4) xyzr
    pc_range: List[float],    # [x_min,y_min,z_min, x_max,y_max,z_max]
    voxel_size: List[float],  # [vx, vy, vz] metres per voxel
    max_points_per_voxel: int = 32,
    max_voxels:           int = 20000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple pillar/voxel grid construction (PointPillars-style).

    Returns:
        voxels      : (V, max_pts, 4)   — padded point features per voxel
        coords      : (V, 3)            — voxel grid coordinates (z, y, x)
        num_points  : (V,)              — actual point count per voxel
    """
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    vx, vy, vz = voxel_size

    grid_x = int((x_max - x_min) / vx)
    grid_y = int((y_max - y_min) / vy)
    grid_z = int((z_max - z_min) / vz)

    # Filter points to pc_range
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    points = points[mask]

    if len(points) == 0:
        return (np.zeros((0, max_points_per_voxel, 4), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros(0, dtype=np.int32))

    # Voxel index for each point
    xi = np.floor((points[:, 0] - x_min) / vx).astype(int).clip(0, grid_x - 1)
    yi = np.floor((points[:, 1] - y_min) / vy).astype(int).clip(0, grid_y - 1)
    zi = np.floor((points[:, 2] - z_min) / vz).astype(int).clip(0, grid_z - 1)

    voxel_idx = zi * grid_y * grid_x + yi * grid_x + xi
    order     = np.argsort(voxel_idx)
    points    = points[order]
    voxel_idx = voxel_idx[order]

    # Aggregate points per voxel
    unique_voxels, inverse, counts = np.unique(
        voxel_idx, return_inverse=True, return_counts=True
    )

    num_voxels = min(len(unique_voxels), max_voxels)
    voxels     = np.zeros((num_voxels, max_points_per_voxel, 4), dtype=np.float32)
    coords     = np.zeros((num_voxels, 3), dtype=np.int32)
    num_pts    = np.zeros(num_voxels, dtype=np.int32)

    point_start = 0
    for v_i in range(num_voxels):
        cnt  = counts[v_i]
        pts  = points[point_start : point_start + cnt]
        n    = min(cnt, max_points_per_voxel)
        voxels[v_i, :n] = pts[:n, :4]
        num_pts[v_i]    = n

        flat = unique_voxels[v_i]
        z_c  = flat // (grid_y * grid_x)
        y_c  = (flat % (grid_y * grid_x)) // grid_x
        x_c  = flat % grid_x
        coords[v_i] = [z_c, y_c, x_c]

        point_start += cnt

    return voxels, coords, num_pts