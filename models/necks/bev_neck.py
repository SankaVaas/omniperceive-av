"""
BEV Neck — Perspective-View → Bird's-Eye-View Projection
=========================================================
Lifts multi-camera image features from perspective view (PV)
into a unified Bird's-Eye-View (BEV) grid.

We implement a simplified Lift-Splat-Shoot (LSS) style projection
(Philion & Fidler, NeurIPS 2020) using a depth distribution over
a discrete depth bin range. For each image pixel, the feature is
"splatted" to the BEV voxel it falls in, weighted by the predicted
depth probability.

Pipeline:
  FPN features (B, C, H_feat, W_feat)
       ↓  depth head: predict D-bin depth distribution per pixel
  Frustum (B, D, H_feat, W_feat, C)  — lift
       ↓  unproject using camera intrinsics + extrinsics
  Voxel grid (B, C, X, Y, Z)         — splat
       ↓  collapse Z dimension
  BEV feature map (B, C_bev, bev_h, bev_w) — shoot

For single-camera KITTI we use only CAM_FRONT.
For nuScenes multi-camera, all 6 views are projected and max-pooled.

Reference: Philion & Fidler "Lift, Splat, Shoot: Encoding Images from
           Arbitrary Camera Rigs by Implicitly Unprojecting to 3D"
           NeurIPS 2020. https://arxiv.org/abs/2008.05711
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DepthDistributionHead(nn.Module):
    """
    Predicts a categorical depth distribution over D bins for each pixel.

    This lightweight head branches off the FPN P2 feature and outputs
    a softmax distribution: which depth bin does this pixel fall in?

    Args:
        in_channels : FPN feature channels.
        num_depth_bins : number of discrete depth bins D.
        hidden_channels : intermediate conv channels.
    """

    def __init__(self, in_channels: int, num_depth_bins: int, hidden_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_depth_bins, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W) FPN feature map
        Returns:
            depth_dist : (B, D, H, W) softmax depth distribution
        """
        return self.net(x).softmax(dim=1)


class BEVPooling(nn.Module):
    """
    Splat-pool lifted frustum features into a BEV grid.

    For each 3D point (x, y, z) in ego frame, we find which BEV
    pixel (i, j) it falls in and accumulate the feature there.
    Uses a differentiable scatter (torch.scatter_add).

    Args:
        bev_h, bev_w : BEV grid spatial dimensions.
        pc_range     : [x_min, y_min, z_min, x_max, y_max, z_max] in metres.
    """

    def __init__(self, bev_h: int, bev_w: int, pc_range: List[float]):
        super().__init__()
        self.bev_h    = bev_h
        self.bev_w    = bev_w
        self.pc_range = pc_range

    def forward(
        self,
        points_3d: torch.Tensor,   # (B, N, 3) in ego frame
        features:  torch.Tensor,   # (B, N, C)
    ) -> torch.Tensor:
        """
        Returns:
            bev : (B, C, bev_h, bev_w)
        """
        B, N, C = features.shape
        device  = features.device

        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        x_res = self.bev_w / (x_max - x_min)
        y_res = self.bev_h / (y_max - y_min)

        # BEV pixel index for each 3D point
        px = ((points_3d[:, :, 0] - x_min) * x_res).long()   # (B, N)
        py = ((points_3d[:, :, 1] - y_min) * y_res).long()   # (B, N)

        # Validity mask: inside BEV range
        valid = (
            (px >= 0) & (px < self.bev_w) &
            (py >= 0) & (py < self.bev_h) &
            (points_3d[:, :, 2] > self.pc_range[2]) &
            (points_3d[:, :, 2] < self.pc_range[5])
        )  # (B, N)

        # Flatten BEV pixel index
        flat_idx = py * self.bev_w + px   # (B, N)
        flat_idx = flat_idx.clamp(0, self.bev_h * self.bev_w - 1)

        # Scatter-add: accumulate features at BEV pixels
        bev_flat = torch.zeros(B, self.bev_h * self.bev_w, C, device=device)
        idx_exp  = flat_idx.unsqueeze(-1).expand(-1, -1, C)   # (B, N, C)

        # Zero out invalid points before scatter
        feat_masked = features * valid.unsqueeze(-1).float()
        bev_flat.scatter_add_(1, idx_exp, feat_masked)

        # Count hits per cell for mean pooling
        counts = torch.zeros(B, self.bev_h * self.bev_w, 1, device=device)
        counts.scatter_add_(1, flat_idx.unsqueeze(-1),
                            valid.unsqueeze(-1).float())
        bev_flat = bev_flat / (counts + 1e-6)

        return bev_flat.view(B, self.bev_h, self.bev_w, C).permute(0, 3, 1, 2).contiguous()


class FrustumLift(nn.Module):
    """
    Lift image features to a 3D point cloud (frustum) using
    a predicted depth distribution over discrete depth bins.

    For each image pixel (u, v) and depth bin d_k:
        3D point = K^-1 @ [u, v, 1] * d_k  (in camera frame)
        → transform to ego frame via cam→ego extrinsics

    Args:
        depth_min   : minimum depth in metres.
        depth_max   : maximum depth in metres.
        num_depth_bins : D — number of discrete depth bins.
        in_channels : feature channels.
        out_channels : output feature channels per 3D point.
    """

    def __init__(
        self,
        depth_min:      float,
        depth_max:      float,
        num_depth_bins: int,
        in_channels:    int,
        out_channels:   int,
    ):
        super().__init__()
        self.D = num_depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max

        # Depth bin centres (uniform linear spacing)
        depth_bins = torch.linspace(depth_min, depth_max, num_depth_bins)
        self.register_buffer("depth_bins", depth_bins)   # (D,)

        # Lightweight feature projection
        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.depth_head = DepthDistributionHead(in_channels, num_depth_bins)

    def forward(
        self,
        feat: torch.Tensor,           # (B, C, H_f, W_f) — FPN P2
        K:    torch.Tensor,           # (B, 3, 3) intrinsics
        cam2ego: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # cam2ego: (rot (B,3,3), t (B,3)) or None → stays in camera frame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            points_3d : (B, D*H_f*W_f, 3) in ego frame
            features  : (B, D*H_f*W_f, out_channels) feature per point
        """
        B, C, H_f, W_f = feat.shape

        # Predict depth distribution
        depth_dist = self.depth_head(feat)   # (B, D, H_f, W_f)
        feat_proj  = self.feat_proj(feat)    # (B, out_C, H_f, W_f)
        out_C      = feat_proj.shape[1]

        # Build pixel grid
        ys, xs = torch.meshgrid(
            torch.arange(H_f, device=feat.device, dtype=torch.float32),
            torch.arange(W_f, device=feat.device, dtype=torch.float32),
            indexing="ij",
        )   # (H_f, W_f)

        # Unproject pixel coords to camera rays (normalised)
        # K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        fx = K[:, 0, 0].view(B, 1, 1)   # (B,1,1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)

        ray_x = (xs.unsqueeze(0) - cx) / fx    # (B, H_f, W_f)
        ray_y = (ys.unsqueeze(0) - cy) / fy    # (B, H_f, W_f)
        ray_z = torch.ones_like(ray_x)          # (B, H_f, W_f)

        # Expand rays over depth bins: (B, D, H_f, W_f, 3)
        D = self.D
        d_bins = self.depth_bins.view(1, D, 1, 1)    # (1, D, 1, 1)

        pts_x = (ray_x.unsqueeze(1) * d_bins).unsqueeze(-1)   # (B, D, H_f, W_f, 1)
        pts_y = (ray_y.unsqueeze(1) * d_bins).unsqueeze(-1)
        pts_z = (ray_z.unsqueeze(1) * d_bins).unsqueeze(-1)

        pts_cam = torch.cat([pts_x, pts_y, pts_z], dim=-1)    # (B, D, H_f, W_f, 3)

        # Transform to ego frame if extrinsics provided
        if cam2ego is not None:
            rot, t = cam2ego    # rot: (B,3,3), t: (B,3)
            # pts_cam: (B, D*H_f*W_f, 3)
            N_pts = D * H_f * W_f
            pts_flat = pts_cam.view(B, N_pts, 3)                       # (B, N, 3)
            pts_ego  = (rot @ pts_flat.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)
        else:
            pts_ego = pts_cam.view(B, D * H_f * W_f, 3)

        # Weight features by depth probability: (B, D, H_f, W_f) × (B, out_C, H_f, W_f)
        # → (B, D, H_f, W_f, out_C)
        depth_w = depth_dist.permute(0, 2, 3, 1).unsqueeze(-1)        # (B, H_f, W_f, D, 1)
        feat_exp = feat_proj.permute(0, 2, 3, 1).unsqueeze(3)         # (B, H_f, W_f, 1, out_C)
        weighted = (feat_exp * depth_w).permute(0, 3, 1, 2, 4).contiguous()  # (B, D, H_f, W_f, out_C)
        feat_flat = weighted.view(B, D * H_f * W_f, out_C)                  # (B, N, out_C)

        return pts_ego, feat_flat


class BEVNeck(nn.Module):
    """
    BEV Neck: lifts FPN features into a unified BEV grid.

    For single-camera (KITTI): uses CAM_FRONT FPN P2 feature only.
    For multi-camera (nuScenes): each camera's features are lifted
    separately and max-pooled into the shared BEV grid.

    The BEV feature map is then passed through a lightweight 2D CNN
    to refine cross-view blending artefacts before reaching the heads.

    Args:
        in_channels     : FPN output channels (from neck).
        bev_h, bev_w    : BEV grid size in pixels.
        pc_range        : [x_min, y_min, z_min, x_max, y_max, z_max]
        num_depth_bins  : D discrete depth bins for the depth distribution.
        depth_min/max   : depth range in metres.
        out_channels    : BEV feature channels after refinement conv.
        num_cameras     : 1 for KITTI; 6 for nuScenes.
    """

    def __init__(
        self,
        in_channels:    int,
        bev_h:          int,
        bev_w:          int,
        pc_range:       List[float],
        num_depth_bins: int  = 64,
        depth_min:      float = 0.5,
        depth_max:      float = 80.0,
        out_channels:   int  = 256,
        num_cameras:    int  = 1,
        **kwargs,   # absorb extra cfg keys (e.g. use_deformable_attention)
    ):
        super().__init__()
        self.bev_h       = bev_h
        self.bev_w       = bev_w
        self.pc_range    = pc_range
        self.num_cameras = num_cameras
        self.out_channels = out_channels

        # ── LSS modules ──────────────────────────────────────────────────
        self.lift = FrustumLift(
            depth_min=depth_min,
            depth_max=depth_max,
            num_depth_bins=num_depth_bins,
            in_channels=in_channels,
            out_channels=in_channels,   # project to same dim, refine after
        )
        self.pool = BEVPooling(bev_h, bev_w, pc_range)

        # ── BEV refinement: 2D ConvBNReLU stack ─────────────────────────
        # Bridges the gap between noisy scatter output and clean feature map
        self.bev_refine = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # ── Multi-scale BEV output (for detection + lane heads) ──────────
        # Produce 2 BEV scales from the refined feature map
        self.bev_ds = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _lift_one_camera(
        self,
        feat:    torch.Tensor,           # (B, C, H_f, W_f)
        K:       torch.Tensor,           # (B, 3, 3)
        cam2ego: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """Lift one camera view and pool to BEV. Returns (B, C, bev_h, bev_w)."""
        pts, feats = self.lift(feat, K, cam2ego)
        return self.pool(pts, feats)

    def forward(
        self,
        fpn_feats: List[torch.Tensor],
        K:         Optional[torch.Tensor] = None,
        cam2ego:   Optional[Tuple]        = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            fpn_feats : list of FPN feature maps [P2(fine), P3, P4, P5].
                        We use P2 (finest) for lifting.
            K         : camera intrinsics.
                        Single-cam: (B, 3, 3)
                        Multi-cam:  (B, num_cams, 3, 3)
            cam2ego   : (rot, t) camera-to-ego extrinsics.
                        Single-cam: rot (B,3,3), t (B,3)
                        Multi-cam:  rot (B, num_cams, 3, 3), t (B, num_cams, 3)

        Returns:
            List of BEV feature maps [bev_fine, bev_coarse]:
              bev_fine   : (B, out_channels, bev_h,   bev_w)
              bev_coarse : (B, out_channels, bev_h//2, bev_w//2)

        Note on missing calibration (K=None):
            When K is not provided (e.g. early debug runs), the neck falls
            back to a simple global average pool + spatial reshape to BEV size.
            This lets the rest of the model run for shape-testing without data.
        """
        feat_p2 = fpn_feats[0]   # finest FPN level (B, C, H/8, W/8)
        B, C, H_f, W_f = feat_p2.shape

        if K is None:
            # ── Fallback: no calibration provided ────────────────────────
            # Resize P2 directly to BEV size (purely spatial, no 3D lift)
            bev = F.interpolate(
                feat_p2, size=(self.bev_h, self.bev_w),
                mode="bilinear", align_corners=False
            )
        elif self.num_cameras == 1:
            # ── Single camera (KITTI) ─────────────────────────────────────
            # K shape: (B, 3, 3)
            bev = self._lift_one_camera(feat_p2, K, cam2ego)
        else:
            # ── Multi-camera (nuScenes) ───────────────────────────────────
            # K shape: (B, num_cams, 3, 3)
            # cam2ego: rot (B, num_cams, 3, 3), t (B, num_cams, 3)
            bev_views = []
            for cam_i in range(self.num_cameras):
                K_i = K[:, cam_i]                          # (B, 3, 3)
                ce_i = None
                if cam2ego is not None:
                    rot_i = cam2ego[0][:, cam_i]           # (B, 3, 3)
                    t_i   = cam2ego[1][:, cam_i]           # (B, 3)
                    ce_i  = (rot_i, t_i)
                bev_i = self._lift_one_camera(feat_p2, K_i, ce_i)
                bev_views.append(bev_i)

            # Max-pool across camera views: each view contributes where it has
            # the strongest feature (non-overlapping regions → clean fusion)
            bev = torch.stack(bev_views, dim=0).max(dim=0).values

        # ── Refinement CNN ────────────────────────────────────────────────
        bev_fine   = self.bev_refine(bev)         # (B, out_C, bev_h, bev_w)
        bev_coarse = self.bev_ds(bev_fine)        # (B, out_C, bev_h//2, bev_w//2)

        return [bev_fine, bev_coarse]


def build_bev_neck(cfg: dict) -> BEVNeck:
    """Instantiate BEVNeck from config dict (bev_neck section of YAML)."""
    return BEVNeck(
        in_channels=cfg["in_channels"],
        bev_h=cfg["bev_h"],
        bev_w=cfg["bev_w"],
        pc_range=cfg["pc_range"],
        num_depth_bins=cfg.get("num_depth_bins", 64),
        depth_min=cfg.get("depth_min", 0.5),
        depth_max=cfg.get("depth_max", 80.0),
        out_channels=cfg.get("out_channels", cfg["in_channels"]),
        num_cameras=cfg.get("num_cameras", 1),
    )