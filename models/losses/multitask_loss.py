"""
Multi-Task Loss Orchestrator
============================
Computes individual task losses and combines them via
HomoscedasticUncertaintyLoss (Kendall & Gal, CVPR 2018).

Task losses:
  Detection   : CenterPoint-style modified focal + L1 offset/wh regression
  Lane        : Smooth-L1 on polynomial coefficients + BCE confidence
  Depth       : Self-supervised Monodepth2 (photometric + SSIM + smoothness)
  Segmentation: Dice + Binary Cross-Entropy

Key wiring notes:
  - outputs["det"]   is a dict  {heatmap, offset, wh, z_center}
  - outputs["lane"]  is a dict  {lane_coeffs, lane_conf, lane_yrange, anchor_pos}
  - outputs["depth"] is a dict  {depth, disp_scales, warped_imgs}
  - outputs["seg"]   is a tensor (B, num_classes, H_p2, W_p2) — raw logits
  - Seg GT is upsampled to match pred spatial size before loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from models.losses.uncertainty_loss import HomoscedasticUncertaintyLoss
from models.losses.depth_loss       import SelfSupervisedDepthLoss


class MultiTaskUncertaintyLoss(nn.Module):
    """
    Orchestrates all four task losses with learnable uncertainty weighting.

    Args:
        num_tasks   : must be 4.
        focal_alpha : focal loss exponent on positive predictions (det).
        focal_gamma : focal loss exponent on negative down-weighting (det).
        cfg         : optional loss sub-config from YAML (overrides defaults).
    """

    def __init__(
        self,
        num_tasks:   int   = 4,
        focal_alpha: float = 2.0,
        focal_gamma: float = 4.0,
        cfg:         dict  = None,
    ):
        super().__init__()
        assert num_tasks == 4

        cfg = cfg or {}
        init_log_sigma = cfg.get("uncertainty_init_log_sigma", -0.5)

        self.uncertainty  = HomoscedasticUncertaintyLoss(
            num_tasks=num_tasks, init_log_sigma=init_log_sigma
        )
        self.depth_loss   = SelfSupervisedDepthLoss(
            ssim_weight=cfg.get("ssim_weight",   0.85),
            smooth_weight=cfg.get("smooth_weight", 1e-3),
        )
        self.focal_alpha  = cfg.get("focal_alpha", focal_alpha)
        self.focal_gamma  = cfg.get("focal_gamma", focal_gamma)

    # ─────────────────────────────────────────────────────────────────────
    # Detection — CenterPoint modified focal loss
    # ─────────────────────────────────────────────────────────────────────
    def _detection_loss(
        self,
        pred: Dict[str, torch.Tensor],
        gt:   Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Heatmap: modified focal loss (CornerNet / CenterPoint formulation).
        Offset + WH: L1, evaluated only at GT peak locations.
        """
        hm_pred = pred["heatmap"]          # (B, C, H_bev, W_bev) — already sigmoid
        hm_gt   = gt["heatmap"].to(hm_pred)   # (B, C, H_bev, W_bev)

        # ── Resize if BEV head resolution differs from GT ─────────────────
        if hm_pred.shape != hm_gt.shape:
            hm_gt = F.interpolate(hm_gt, size=hm_pred.shape[2:], mode="nearest")

        pos_mask    = hm_gt.eq(1.0).float()
        neg_weights = torch.pow(1.0 - hm_gt, self.focal_gamma)

        pos_loss = (
            -torch.log(hm_pred.clamp(min=1e-6))
            * torch.pow(1.0 - hm_pred, self.focal_alpha)
            * pos_mask
        )
        neg_loss = (
            -torch.log((1.0 - hm_pred).clamp(min=1e-6))
            * torch.pow(hm_pred, self.focal_alpha)
            * neg_weights
            * (1.0 - pos_mask)
        )

        num_pos = pos_mask.sum().clamp(min=1.0)
        focal   = (pos_loss.sum() + neg_loss.sum()) / num_pos

        # Regression at positive peaks only
        device = focal.device
        offset_loss = torch.tensor(0.0, device=device)
        wh_loss     = torch.tensor(0.0, device=device)

        if pos_mask.sum() > 0:
            # pos_mask is (B,C,H,W) — broadcast across offset/wh channels
            # Take any-class peak mask: (B, H, W)
            peak = pos_mask.any(dim=1)                  # (B, H_bev, W_bev)
            peak_exp = peak.unsqueeze(1).expand_as(pred["offset"])  # (B,2,H,W)

            if peak_exp.any():
                offset_loss = F.l1_loss(
                    pred["offset"][peak_exp],
                    gt["offset"].to(device)[peak_exp],
                )
                wh_loss = F.l1_loss(
                    pred["wh"][peak_exp],
                    gt["wh"].to(device)[peak_exp],
                )

        return focal + 0.1 * offset_loss + 0.1 * wh_loss

    # ─────────────────────────────────────────────────────────────────────
    # Lane — Smooth-L1 coefficients + BCE confidence
    # ─────────────────────────────────────────────────────────────────────
    def _lane_loss(
        self,
        pred: Dict[str, torch.Tensor],
        gt:   Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        pred keys: lane_coeffs (B, NL, K), lane_conf (B, NL)
        gt keys  : lane_coeffs (B, NL, K), lane_conf  (B, NL) — 0/1 presence
        """
        gt_conf   = gt["lane_conf"].to(pred["lane_conf"])    # (B, NL)
        gt_coeffs = gt["lane_coeffs"].to(pred["lane_coeffs"])

        conf_loss  = F.binary_cross_entropy_with_logits(pred["lane_conf"], gt_conf)

        valid = gt_conf > 0.5
        coeff_loss = torch.tensor(0.0, device=pred["lane_coeffs"].device)
        if valid.any():
            coeff_loss = F.smooth_l1_loss(
                pred["lane_coeffs"][valid],
                gt_coeffs[valid],
                beta=0.1,
            )

        return conf_loss + coeff_loss

    # ─────────────────────────────────────────────────────────────────────
    # Segmentation — Dice + BCE
    # ─────────────────────────────────────────────────────────────────────
    def _seg_loss(
        self,
        pred_logits: torch.Tensor,    # (B, 1, H_p2, W_p2)
        gt_mask:     torch.Tensor,    # (B, 1, H_gt, W_gt)  or (B, H, W)
    ) -> torch.Tensor:
        # Ensure GT has channel dim
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.unsqueeze(1)
        gt_mask = gt_mask.to(pred_logits)

        # Resize GT to match pred resolution (P2 = H/8 × W/8)
        if gt_mask.shape[2:] != pred_logits.shape[2:]:
            gt_mask = F.interpolate(
                gt_mask, size=pred_logits.shape[2:], mode="nearest"
            )

        pred_sig = torch.sigmoid(pred_logits)
        bce  = F.binary_cross_entropy_with_logits(pred_logits, gt_mask)

        inter = (pred_sig * gt_mask).sum(dim=(2, 3))                 # (B, 1)
        union = pred_sig.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3))   # (B, 1)
        dice  = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)

        return bce + dice.mean()

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs : model prediction dict with keys det, lane, depth, seg.
            targets : GT dict from dataset / collate_fn.

        Returns:
            dict with scalar 'loss' (backward-able) + per-task losses + log_vars.
        """
        l_det   = self._detection_loss(outputs["det"],  targets)
        l_lane  = self._lane_loss(outputs["lane"],      targets)
        l_depth = self.depth_loss(outputs["depth"],     targets)
        l_seg   = self._seg_loss(outputs["seg"],        targets["seg_mask"])

        total, log_vars = self.uncertainty([l_det, l_lane, l_depth, l_seg])

        return {
            "loss":       total,
            "loss_det":   l_det.detach(),
            "loss_lane":  l_lane.detach(),
            "loss_depth": l_depth.detach(),
            "loss_seg":   l_seg.detach(),
            "log_vars":   log_vars,      # (4,) sigma per task — log to TensorBoard
        }