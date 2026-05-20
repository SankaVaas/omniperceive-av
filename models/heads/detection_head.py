"""
CenterPoint-Style 3D Detection Head
=====================================
Produces per-class Gaussian heatmaps on the BEV grid, plus regression
offsets for sub-voxel localisation and object dimensions.

Outputs:
    heatmap  : (B, num_classes, H_bev, W_bev)  — sigmoid, range [0,1]
    offset   : (B, 2, H_bev, W_bev)            — sub-voxel x,y offset
    wh       : (B, 2, H_bev, W_bev)            — log-space w, h (BEV footprint)
    z_center : (B, 1, H_bev, W_bev)            — height regression

Reference: Yin et al. "Center-based 3D Object Detection and Tracking" — CVPR 2021
"""

import torch
import torch.nn as nn
from typing import Dict


class SeparableConv(nn.Sequential):
    """Depthwise-separable conv block: DWConv → BN → ReLU → PWConv → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int, bias: bool = False):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DetectionHead(nn.Module):
    """
    Shared feature → multiple regression/classification outputs.

    Args:
        cfg          : head config dict (num_classes, hidden_channels, …)
        in_channels  : feature channels from BEV neck output.
    """

    def __init__(self, cfg: dict, in_channels: int):
        super().__init__()
        hidden    = cfg.get("hidden_channels", 64)
        num_cls   = cfg.get("num_classes", 10)

        self.shared = nn.Sequential(
            SeparableConv(in_channels, hidden),
            SeparableConv(hidden, hidden),
        )

        # Each output head: conv → (optional norm) → output
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(hidden, num_cls, 3, padding=1),
            # NOTE: do NOT apply sigmoid here — applied in loss / postprocess
        )
        self.offset_head   = nn.Conv2d(hidden, 2, 3, padding=1)
        self.wh_head       = nn.Conv2d(hidden, 2, 3, padding=1)
        self.z_center_head = nn.Conv2d(hidden, 1, 3, padding=1)

        # Focal loss: initialise heatmap bias to -log((1-pi)/pi), pi=0.01
        nn.init.constant_(self.heatmap_head[0].bias, -4.6)

    def forward(self, bev_feats: list) -> Dict[str, torch.Tensor]:
        """
        Args:
            bev_feats : list of BEV feature maps; uses the highest-resolution.
        Returns:
            dict with keys: heatmap, offset, wh, z_center
        """
        x = bev_feats[0]          # highest-res BEV feature map
        x = self.shared(x)

        return {
            "heatmap":  torch.sigmoid(self.heatmap_head(x)),
            "offset":   self.offset_head(x),
            "wh":       self.wh_head(x),
            "z_center": self.z_center_head(x),
        }

    @torch.no_grad()
    def decode(
        self,
        preds: Dict[str, torch.Tensor],
        score_thresh: float = 0.3,
        max_detections: int = 500,
        nms_radius: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """
        Greedy circle-NMS decoding on the heatmap (no box IoU needed in BEV).

        Args:
            preds          : output dict from forward()
            score_thresh   : minimum confidence to keep a detection
            max_detections : hard cap on returned boxes
            nms_radius     : radius (in BEV pixels) for peak suppression

        Returns:
            dict: scores (N,), classes (N,), bev_boxes (N,4), z_centers (N,)
        """
        heatmap = preds["heatmap"]       # (B, C, H, W)  — already sigmoid
        B, C, H, W = heatmap.shape

        # Max-pool NMS: suppress non-peak pixels
        heatmap_max = nn.functional.max_pool2d(
            heatmap, kernel_size=2 * nms_radius + 1,
            stride=1, padding=nms_radius
        )
        peak_mask = (heatmap == heatmap_max) & (heatmap > score_thresh)

        # Gather top-K peaks across all classes
        scores, classes, ys, xs = [], [], [], []
        for b in range(B):
            for c in range(C):
                idx = peak_mask[b, c].nonzero(as_tuple=False)  # (K, 2)
                if idx.numel() == 0:
                    continue
                s = heatmap[b, c][idx[:, 0], idx[:, 1]]
                # Keep top max_detections by score
                topk = s.topk(min(max_detections, s.numel()))
                scores.append(topk.values)
                classes.append(torch.full_like(topk.values, c, dtype=torch.long))
                ys.append(idx[topk.indices, 0].float())
                xs.append(idx[topk.indices, 1].float())

        if not scores:
            return {"scores": torch.empty(0), "classes": torch.empty(0, dtype=torch.long),
                    "bev_boxes": torch.empty(0, 4), "z_centers": torch.empty(0)}

        scores  = torch.cat(scores)
        classes = torch.cat(classes)
        ys      = torch.cat(ys)
        xs      = torch.cat(xs)

        # Refine with sub-pixel offsets
        off = preds["offset"][0]   # (2, H, W)
        wh  = preds["wh"][0]       # (2, H, W)
        zi  = preds["z_center"][0] # (1, H, W)

        yi = ys.long().clamp(0, H - 1)
        xi = xs.long().clamp(0, W - 1)

        cx = xs + off[0, yi, xi]
        cy = ys + off[1, yi, xi]
        bw = wh[0, yi, xi].exp()
        bh = wh[1, yi, xi].exp()
        zc = zi[0, yi, xi]

        bev_boxes = torch.stack([cx - bw / 2, cy - bh / 2,
                                 cx + bw / 2, cy + bh / 2], dim=1)

        return {"scores": scores, "classes": classes,
                "bev_boxes": bev_boxes, "z_centers": zc}
