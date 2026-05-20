"""
Multi-Task Loss Orchestrator
============================
Computes individual task losses and combines them via HomoscedasticUncertaintyLoss.

Task losses used:
  - Detection   : Focal loss on heatmap + L1 on offsets/dims (CenterPoint-style)
  - Lane        : Smooth-L1 on polynomial coefficients + BCE on lane confidence
  - Depth       : Self-supervised photometric + SSIM + edge-aware smoothness
  - Segmentation: Dice + Binary Cross-Entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from models.losses.uncertainty_loss import HomoscedasticUncertaintyLoss
from models.losses.depth_loss import SelfSupervisedDepthLoss


class MultiTaskUncertaintyLoss(nn.Module):
    """
    Orchestrates all four task losses with learnable uncertainty weighting.

    Args:
        num_tasks (int): Must be 4. Defined explicitly for clarity.
        focal_alpha (float): Focal loss alpha for detection heatmap.
        focal_gamma (float): Focal loss gamma for detection heatmap.
    """

    def __init__(self, num_tasks: int = 4, focal_alpha: float = 2.0, focal_gamma: float = 4.0):
        super().__init__()
        assert num_tasks == 4
        self.uncertainty = HomoscedasticUncertaintyLoss(num_tasks=num_tasks)
        self.depth_loss  = SelfSupervisedDepthLoss()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    # ── Detection: CenterPoint-style focal loss ───────────────────────────
    def _detection_loss(self, pred: Dict, gt: Dict) -> torch.Tensor:
        heatmap_pred = pred["heatmap"]   # (B, C, H, W) — sigmoid applied in head
        heatmap_gt   = gt["heatmap"]     # (B, C, H, W) — Gaussian-rendered GT

        # Modified focal loss (CenterNet / CenterPoint formulation)
        pos_mask = heatmap_gt.eq(1.0).float()
        neg_mask = heatmap_gt.lt(1.0).float()

        neg_weights = torch.pow(1.0 - heatmap_gt, self.focal_gamma)

        pos_loss = -torch.log(heatmap_pred.clamp(min=1e-6)) * \
                   torch.pow(1.0 - heatmap_pred, self.focal_alpha) * pos_mask
        neg_loss = -torch.log((1.0 - heatmap_pred).clamp(min=1e-6)) * \
                   torch.pow(heatmap_pred, self.focal_alpha) * neg_weights * neg_mask

        num_pos = pos_mask.sum().clamp(min=1)
        focal   = (pos_loss.sum() + neg_loss.sum()) / num_pos

        # Regression losses (only at positive locations)
        offset_loss = F.l1_loss(
            pred["offset"][pos_mask.bool()], gt["offset"][pos_mask.bool()], reduction="mean"
        ) if pos_mask.sum() > 0 else torch.tensor(0.0, device=focal.device)

        wh_loss = F.l1_loss(
            pred["wh"][pos_mask.bool()], gt["wh"][pos_mask.bool()], reduction="mean"
        ) if pos_mask.sum() > 0 else torch.tensor(0.0, device=focal.device)

        return focal + 0.1 * offset_loss + 0.1 * wh_loss

    # ── Lane: polynomial coefficient regression + confidence ─────────────
    def _lane_loss(self, pred: Dict, gt: Dict) -> torch.Tensor:
        conf_gt   = gt["lane_conf"]              # (B, max_lanes)
        valid     = conf_gt > 0.5

        coeff_loss = torch.tensor(0.0, device=pred["lane_coeffs"].device)
        if valid.any():
            coeff_loss = F.smooth_l1_loss(
                pred["lane_coeffs"][valid], gt["lane_coeffs"][valid]
            )
        conf_loss = F.binary_cross_entropy_with_logits(
            pred["lane_conf"], conf_gt.float()
        )
        return coeff_loss + conf_loss

    # ── Segmentation: Dice + BCE combo ───────────────────────────────────
    def _seg_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_sig = torch.sigmoid(pred)
        bce  = F.binary_cross_entropy_with_logits(pred, gt.float())

        # Dice loss
        inter = (pred_sig * gt.float()).sum(dim=(2, 3))
        dice  = 1.0 - (2.0 * inter + 1.0) / \
                      (pred_sig.sum(dim=(2, 3)) + gt.float().sum(dim=(2, 3)) + 1.0)

        return bce + dice.mean()

    def forward(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        l_det  = self._detection_loss(outputs["det"],   targets)
        l_lane = self._lane_loss(outputs["lane"],       targets)
        l_depth = self.depth_loss(outputs["depth"],     targets)
        l_seg  = self._seg_loss(outputs["seg"],         targets["seg_mask"])

        total, log_vars = self.uncertainty([l_det, l_lane, l_depth, l_seg])

        return {
            "loss":       total,
            "loss_det":   l_det.detach(),
            "loss_lane":  l_lane.detach(),
            "loss_depth": l_depth.detach(),
            "loss_seg":   l_seg.detach(),
            "log_vars":   log_vars,          # (4,) — log to TensorBoard
        }
