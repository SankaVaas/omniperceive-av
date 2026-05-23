"""
BEV Polynomial Lane Detection Head
=====================================
Detects lanes directly in Bird's-Eye-View space as polynomial curves.

Design rationale:
  - Working in BEV avoids the perspective distortion of image-space lane
    detection (e.g. SCNN, LaneATT), making polynomial regression well-posed.
  - Each lane is modelled as a cubic polynomial: x = a·y³ + b·y² + c·y + d
    where y is the longitudinal BEV axis and x is lateral displacement.
  - We predict per-anchor polynomial coefficients + confidence score.
  - Anchors are evenly spaced across BEV width at fixed y-strides.

Architecture:
  BEV feature (B, C, bev_h, bev_w)
      ↓  shared 3×3 conv trunk (separable)
      ├── Coefficient head → (B, num_lanes, poly_degree+1)
      └── Confidence head  → (B, num_lanes)    logits

Post-processing:
  - Sigmoid confidence, threshold at conf_thresh
  - Polynomial evaluated at discrete y-samples for visualisation
  - NMS on lane endpoints to remove duplicates

Reference: inspired by PolyLaneNet (Tabelini et al., ICPR 2021)
           and BEV-LaneDet (Wang et al., CVPR 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class SeparableConvBnRelu(nn.Sequential):
    """Depthwise-separable conv block used throughout the lane head."""
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        pad = dilation
        super().__init__(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, padding=pad, dilation=dilation,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class LaneAnchorGenerator(nn.Module):
    """
    Generates anchor query embeddings positioned across BEV width.

    Each anchor represents a candidate lane starting at a different
    lateral position. The anchor embedding is added to the BEV feature
    before regression, giving the head a positional prior.

    Args:
        num_lanes    : number of lane anchors.
        bev_w        : BEV grid width (pixels).
        embed_dim    : anchor embedding dimension (= hidden_channels).
    """

    def __init__(self, num_lanes: int, bev_w: int, embed_dim: int):
        super().__init__()
        self.num_lanes = num_lanes
        # Learnable anchor positions along BEV width
        self.anchor_x = nn.Parameter(
            torch.linspace(0.1, 0.9, num_lanes)  # normalised [0,1]
        )
        # Per-anchor embedding vector
        self.anchor_embed = nn.Embedding(num_lanes, embed_dim)
        nn.init.normal_(self.anchor_embed.weight, std=0.02)

    def forward(self, bev_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bev_feat : (B, C, bev_h, bev_w)
        Returns:
            anchor_feats : (B, num_lanes, C) — BEV features sampled at anchor x
            anchor_pos   : (num_lanes,) — normalised lateral position
        """
        B, C, bev_h, bev_w = bev_feat.shape

        # Sample BEV feature at each anchor's lateral position
        # Use column-mean to get a 1D lane feature per x position
        bev_col = bev_feat.mean(dim=2)   # (B, C, bev_w)

        # Gather at anchor x positions (interpolate)
        ax = self.anchor_x.clamp(0, 1) * (bev_w - 1)   # pixel coords
        ax_floor = ax.long().clamp(0, bev_w - 2)
        ax_frac  = (ax - ax_floor.float()).view(1, 1, -1)  # (1, 1, num_lanes)

        feat_floor = bev_col[:, :, ax_floor]    # (B, C, num_lanes)
        feat_ceil  = bev_col[:, :, (ax_floor + 1).clamp(max=bev_w-1)]
        anchor_feats = feat_floor + ax_frac * (feat_ceil - feat_floor)  # (B, C, num_lanes)
        anchor_feats = anchor_feats.permute(0, 2, 1)   # (B, num_lanes, C)

        # Add learnable anchor embedding
        idx = torch.arange(self.num_lanes, device=bev_feat.device)
        anchor_feats = anchor_feats + self.anchor_embed(idx).unsqueeze(0)

        return anchor_feats, self.anchor_x


class LaneHead(nn.Module):
    """
    BEV polynomial lane detection head.

    Args:
        cfg         : head config dict with keys:
                        num_lanes     (int)   : max simultaneous lanes (default 6)
                        poly_degree   (int)   : polynomial degree (default 3 = cubic)
                        hidden_channels (int) : trunk feature channels (default 64)
                        anchor_stride (int)   : BEV row stride for anchor pooling
                        conf_thresh   (float) : confidence threshold for NMS (default 0.5)
        in_channels : BEV feature channels from BEVNeck.
    """

    def __init__(self, cfg: dict, in_channels: int):
        super().__init__()
        self.num_lanes   = cfg.get("num_lanes",      6)
        self.poly_degree = cfg.get("poly_degree",    3)
        self.conf_thresh = cfg.get("conf_thresh",    0.5)
        hidden           = cfg.get("hidden_channels", 64)
        bev_w            = cfg.get("bev_w",          128)

        num_coeffs = self.poly_degree + 1   # cubic → 4 coefficients

        # ── Shared trunk (spatial context across full BEV width) ─────────
        self.trunk = nn.Sequential(
            SeparableConvBnRelu(in_channels, hidden),
            SeparableConvBnRelu(hidden,      hidden, dilation=2),
            SeparableConvBnRelu(hidden,      hidden, dilation=4),
        )

        # ── Anchor generator ─────────────────────────────────────────────
        self.anchor_gen = LaneAnchorGenerator(self.num_lanes, bev_w, hidden)

        # ── Regression heads ─────────────────────────────────────────────
        # Polynomial coefficient predictor (one MLP per anchor in parallel)
        self.coeff_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_coeffs),
        )

        # Confidence score predictor
        self.conf_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )

        # Y-range predictor: start and end of lane in BEV y-axis (normalised)
        self.yrange_head = nn.Sequential(
            nn.Linear(hidden, 2),
            nn.Sigmoid(),   # outputs in [0,1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Init confidence bias: predict low confidence initially
        nn.init.constant_(self.conf_head[-1].bias, -2.0)

    def forward(self, bev_feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            bev_feats : list from BEVNeck; uses index 0 (finest BEV level).
                        Shape: (B, C, bev_h, bev_w)

        Returns:
            dict with:
              lane_coeffs : (B, num_lanes, poly_degree+1) — polynomial coefficients
              lane_conf   : (B, num_lanes)                — confidence logits
              lane_yrange : (B, num_lanes, 2)             — [y_start, y_end] normalised
              anchor_pos  : (num_lanes,)                  — lateral anchor positions
        """
        x = bev_feats[0]   # (B, C, bev_h, bev_w)

        # Shared feature extraction
        x = self.trunk(x)                                # (B, hidden, bev_h, bev_w)

        # Sample per-anchor features
        anchor_feats, anchor_pos = self.anchor_gen(x)   # (B, num_lanes, hidden)

        # Predict per-anchor outputs (batch × num_lanes × dim → flatten → predict)
        B, NL, H = anchor_feats.shape
        feats_flat = anchor_feats.contiguous().view(B * NL, H)

        coeffs  = self.coeff_head(feats_flat).view(B, NL, -1)   # (B, NL, num_coeffs)
        conf    = self.conf_head(feats_flat).view(B, NL)         # (B, NL)
        yrange  = self.yrange_head(feats_flat).view(B, NL, 2)   # (B, NL, 2)

        return {
            "lane_coeffs": coeffs,
            "lane_conf":   conf,
            "lane_yrange": yrange,
            "anchor_pos":  anchor_pos.detach(),
        }

    @torch.no_grad()
    def decode(
        self,
        preds:      Dict[str, torch.Tensor],
        bev_h:      int,
        bev_w:      int,
        num_y_pts:  int = 72,
    ) -> List[Dict]:
        """
        Post-process head predictions to lane polylines in BEV pixel space.

        Steps:
          1. Threshold lanes by confidence (sigmoid > conf_thresh)
          2. Evaluate cubic polynomial at num_y_pts between y_start and y_end
          3. Return list of lane dicts per image in batch

        Args:
            preds       : output dict from forward()
            bev_h/bev_w : BEV grid size (pixels) for coordinate scaling
            num_y_pts   : number of sample points per lane curve

        Returns:
            List (length B) of lists of lane dicts:
              {"pts": (K, 2) float pixel coords, "conf": float}
        """
        B          = preds["lane_conf"].shape[0]
        conf_sig   = torch.sigmoid(preds["lane_conf"])   # (B, NL)
        coeffs     = preds["lane_coeffs"]                # (B, NL, num_coeffs)
        yrange     = preds["lane_yrange"]                # (B, NL, 2)

        results = []
        for b in range(B):
            lanes = []
            for i in range(self.num_lanes):
                conf = conf_sig[b, i].item()
                if conf < self.conf_thresh:
                    continue

                y0 = yrange[b, i, 0].item()
                y1 = yrange[b, i, 1].item()
                if y1 <= y0:
                    continue

                # Sample y values (normalised → BEV pixels)
                y_norm = torch.linspace(y0, y1, num_y_pts, device=coeffs.device)
                y_px   = y_norm * bev_h

                # Evaluate polynomial: x = a*y^3 + b*y^2 + c*y + d
                c = coeffs[b, i]   # (num_coeffs,)
                x_norm = sum(
                    c[k] * y_norm ** (self.poly_degree - k)
                    for k in range(self.poly_degree + 1)
                )
                x_px = x_norm * bev_w

                pts = torch.stack([x_px, y_px], dim=1)   # (num_y_pts, 2)
                in_bounds = (
                    (pts[:, 0] >= 0) & (pts[:, 0] < bev_w) &
                    (pts[:, 1] >= 0) & (pts[:, 1] < bev_h)
                )
                if in_bounds.sum() < 2:
                    continue

                lanes.append({"pts": pts[in_bounds].cpu(), "conf": conf})

            results.append(lanes)

        return results

    def render_lane_on_bev(
        self,
        lanes:  List[Dict],
        bev_h:  int,
        bev_w:  int,
    ) -> torch.Tensor:
        """
        Utility: render decoded lane points onto a blank BEV canvas.

        Args:
            lanes  : output of decode() for one image (list of lane dicts)
            bev_h/w: BEV grid size

        Returns:
            canvas : (bev_h, bev_w) float32 tensor, lane pixels = 1.0
        """
        canvas = torch.zeros(bev_h, bev_w)
        for lane in lanes:
            pts = lane["pts"].long()
            valid = (
                (pts[:, 0] >= 0) & (pts[:, 0] < bev_w) &
                (pts[:, 1] >= 0) & (pts[:, 1] < bev_h)
            )
            canvas[pts[valid, 1], pts[valid, 0]] = 1.0
        return canvas