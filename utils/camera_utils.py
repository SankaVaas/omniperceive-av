"""
Camera Utilities
=================
Differentiable view synthesis (image warping) for self-supervised depth.

Core operation: given a depth map, camera intrinsics K, and a relative
pose (R, t) from source → target, synthesise what the target frame looks
like from the source frame. The photometric error between the synthesised
and actual target image is the self-supervised training signal.

Pipeline:
  1. Backproject target pixels to 3D using predicted depth + K_target
  2. Transform 3D points to source frame using (R, t)
  3. Project to source image plane using K_source
  4. Sample source image at projected pixel coords (grid_sample)

This is done differentiably so gradients flow back to the depth network.

Reference: Godard et al. "Digging Into Self-Supervised Monocular Depth
           Estimation" (Monodepth2), ICCV 2019.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def backproject_depth(
    depth: torch.Tensor,    # (B, 1, H, W)
    K_inv: torch.Tensor,    # (B, 3, 3)  inverse intrinsics of target cam
) -> torch.Tensor:
    """
    Lift target-frame pixels to 3D camera coordinates using depth.

    Returns:
        cam_points : (B, 4, H*W) homogeneous 3D points in target cam frame
    """
    B, _, H, W = depth.shape
    device = depth.device

    # Pixel grid (u, v, 1)
    u = torch.arange(W, device=device, dtype=torch.float32)
    v = torch.arange(H, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing="ij")              # (H, W)
    ones   = torch.ones_like(uu)
    uv1    = torch.stack([uu, vv, ones], dim=0)               # (3, H, W)
    uv1    = uv1.view(1, 3, -1).expand(B, -1, -1)             # (B, 3, H*W)

    # Backproject: X = K^-1 @ uv1 * depth
    cam_pts = torch.bmm(K_inv, uv1)                           # (B, 3, H*W)
    depth_flat = depth.view(B, 1, H * W)                      # (B, 1, H*W)
    cam_pts = cam_pts * depth_flat                             # scale by depth

    # Append homogeneous coordinate
    ones_hom = torch.ones(B, 1, H * W, device=device)
    return torch.cat([cam_pts, ones_hom], dim=1)              # (B, 4, H*W)


def project_to_image(
    points_3d: torch.Tensor,    # (B, 4, N) homogeneous 3D points
    K:         torch.Tensor,    # (B, 3, 3) intrinsics of source camera
    T:         torch.Tensor,    # (B, 4, 4) transformation: target→source
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform 3D points and project to source image coordinates.

    Returns:
        pix_coords  : (B, H, W, 2) normalised pixel coords in [-1, 1]
                      for use with F.grid_sample.
        valid_mask  : (B, 1, H, W) bool — True where projected point is
                      in front of camera and within image bounds.
    """
    B, _, N = points_3d.shape

    # Transform to source frame: [R|t] @ points
    cam_pts = torch.bmm(T, points_3d)[:, :3, :]   # (B, 3, N)

    # Project: p = K @ X / Z
    eps = 1e-7
    Z   = cam_pts[:, 2:3, :].clamp(min=eps)
    uv  = torch.bmm(K, cam_pts / Z)               # (B, 3, N)

    # Infer H, W from N (assumes points were flattened from H×W grid)
    # We pass H and W via a square-ish heuristic; caller reshapes pix_coords
    u = uv[:, 0, :]   # (B, N)
    v = uv[:, 1, :]   # (B, N)

    return u, v, Z.squeeze(1)   # (B, N) each


def warp_image(
    source_img: torch.Tensor,   # (B, C, H_src, W_src)
    depth:      torch.Tensor,   # (B, 1, H_tgt, W_tgt) — target depth
    K:          torch.Tensor,   # (B, 3, 3) — source (= target here) intrinsics
    T:          torch.Tensor,   # (B, 4, 4) relative pose target→source
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Synthesise the target frame from source using predicted depth + pose.

    Steps:
      1. Backproject target pixels → 3D (using K^-1 and depth)
      2. Transform 3D points to source frame (using T)
      3. Project to source image plane (using K)
      4. Bilinear sample source image at projected coords

    Args:
        source_img : source RGB frame to warp from.
        depth      : predicted depth of the target frame.
        K          : shared camera intrinsic matrix (assuming same camera).
        T          : relative pose: target camera → source camera frame.
                     Shape (B, 4, 4). Obtained from PoseNet during training.

    Returns:
        warped     : (B, C, H, W) synthesised target image.
        valid_mask : (B, 1, H, W) True where the warp is valid.
    """
    B, _, H, W = depth.shape

    # ── Step 1: backproject ───────────────────────────────────────────────
    K_inv      = torch.inverse(K)                         # (B, 3, 3)
    cam_points = backproject_depth(depth, K_inv)          # (B, 4, H*W)

    # ── Step 2-3: transform + project ────────────────────────────────────
    u, v, Z    = project_to_image(cam_points, K, T)       # (B, H*W) each

    # Normalise to [-1, 1] for grid_sample (x: left=-1, right=+1)
    u_norm = (2.0 * u / (W - 1)) - 1.0
    v_norm = (2.0 * v / (H - 1)) - 1.0

    pix_coords = torch.stack([u_norm, v_norm], dim=-1)    # (B, H*W, 2)
    pix_coords = pix_coords.view(B, H, W, 2)              # (B, H, W, 2)

    # ── Step 4: bilinear sample ───────────────────────────────────────────
    warped = F.grid_sample(
        source_img, pix_coords,
        mode="bilinear", padding_mode="border", align_corners=True
    )   # (B, C, H, W)

    # Valid mask: point in front of camera + within image bounds
    valid = (
        (Z.view(B, 1, H, W) > 0) &
        (u_norm.view(B, 1, H, W) >= -1) &
        (u_norm.view(B, 1, H, W) <=  1) &
        (v_norm.view(B, 1, H, W) >= -1) &
        (v_norm.view(B, 1, H, W) <=  1)
    )

    return warped, valid


class PoseNet(torch.nn.Module):
    """
    Lightweight PoseNet: predicts 6-DoF relative pose between frame pairs.
    Takes a (B, 6, H, W) pair (target + source concatenated) and outputs
    (R, t) as axis-angle + translation.

    Used during self-supervised depth training to provide the T matrix
    for warp_image. Not needed at inference time.

    Architecture: ResNet-style encoder → global pool → 6-DoF head.
    """

    def __init__(self, in_channels: int = 6, base_channels: int = 16):
        super().__init__()
        bc = base_channels
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, bc,    7, stride=2, padding=3, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(bc,          bc*2,  5, stride=2, padding=2, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(bc*2,        bc*4,  3, stride=2, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(bc*4,        bc*8,  3, stride=2, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.pose_head = torch.nn.Linear(bc * 8, 6)
        # Small init: start with near-identity pose
        torch.nn.init.normal_(self.pose_head.weight, std=1e-4)
        torch.nn.init.zeros_(self.pose_head.bias)

    def forward(
        self, target: torch.Tensor, source: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target, source : (B, 3, H, W) RGB frames.
        Returns:
            R : (B, 3, 3) rotation matrix
            t : (B, 3)    translation vector
        """
        x    = torch.cat([target, source], dim=1)   # (B, 6, H, W)
        feat = self.encoder(x).flatten(1)            # (B, bc*8)
        pose = self.pose_head(feat)                  # (B, 6)

        # Convert axis-angle to rotation matrix
        R = axis_angle_to_matrix(pose[:, :3] * 0.01)   # scale down
        t = pose[:, 3:]                                  # (B, 3)

        return R, t


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle (B, 3) to rotation matrix (B, 3, 3) via Rodrigues formula.
    """
    B     = aa.shape[0]
    angle = aa.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1)
    axis  = aa / angle                                           # (B, 3) unit vector

    c  = torch.cos(angle).squeeze(1)   # (B,)
    s  = torch.sin(angle).squeeze(1)
    t_ = 1 - c

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    R = torch.stack([
        t_*x*x + c,   t_*x*y - s*z, t_*x*z + s*y,
        t_*x*y + s*z, t_*y*y + c,   t_*y*z - s*x,
        t_*x*z - s*y, t_*y*z + s*x, t_*z*z + c,
    ], dim=1).view(B, 3, 3)

    return R


def build_T_matrix(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Assemble 4×4 homogeneous transformation from R (B,3,3) and t (B,3)."""
    B  = R.shape[0]
    T  = torch.eye(4, device=R.device).unsqueeze(0).expand(B, -1, -1).clone()
    T[:, :3, :3] = R
    T[:, :3,  3] = t
    return T