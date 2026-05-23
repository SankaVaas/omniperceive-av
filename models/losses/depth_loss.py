"""
Self-Supervised Monocular Depth Loss
=====================================
Implements the Monodepth2 loss (Godard et al. 2019):
    L = alpha * L_photometric + beta * L_smoothness

L_photometric = min over source frames of:
    lambda_ssim * SSIM(I_t, I_t->s) + (1 - lambda_ssim) * |I_t - I_t->s|_1

L_smoothness  = edge-aware depth map smoothness

Key trick: per-pixel minimum across synthesized views handles occlusion.
Auto-masking (mu) filters out pixels where the model gives no improvement
over the identity warp (stationary pixels / camera motion).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    """Differentiable Structural Similarity Index (patch-based)."""

    def __init__(self, patch_size: int = 3):
        super().__init__()
        self.mu_x_pool   = nn.AvgPool2d(patch_size, 1, padding=patch_size // 2)
        self.mu_y_pool   = nn.AvgPool2d(patch_size, 1, padding=patch_size // 2)
        self.sig_x_pool  = nn.AvgPool2d(patch_size, 1, padding=patch_size // 2)
        self.sig_y_pool  = nn.AvgPool2d(patch_size, 1, padding=patch_size // 2)
        self.sig_xy_pool = nn.AvgPool2d(patch_size, 1, padding=patch_size // 2)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu_x  = self.mu_x_pool(x)
        mu_y  = self.mu_y_pool(y)
        sig_x  = self.sig_x_pool(x ** 2)  - mu_x ** 2
        sig_y  = self.sig_y_pool(y ** 2)  - mu_y ** 2
        sig_xy = self.sig_xy_pool(x * y)  - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + self.C1) * (2 * sig_xy + self.C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sig_x + sig_y + self.C2)
        return torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1)


class SelfSupervisedDepthLoss(nn.Module):
    """
    Full Monodepth2-style self-supervised depth loss.

    Args:
        ssim_weight   (float): Weight of SSIM vs L1 in photometric loss.
        smooth_weight (float): Edge-aware smoothness regularization weight.
        auto_mask     (bool) : Enable auto-masking of stationary pixels.
    """

    def __init__(
        self,
        ssim_weight:   float = 0.85,
        smooth_weight: float = 1e-3,
        auto_mask:     bool  = True,
    ):
        super().__init__()
        self.ssim          = SSIM()
        self.ssim_weight   = ssim_weight
        self.smooth_weight = smooth_weight
        self.auto_mask     = auto_mask

    def photometric_loss(
        self, pred_img: torch.Tensor, target_img: torch.Tensor
    ) -> torch.Tensor:
        """Per-pixel photometric reconstruction error. Resizes target if needed."""
        if target_img.shape[2:] != pred_img.shape[2:]:
            target_img = F.interpolate(target_img, size=pred_img.shape[2:],
                                       mode="bilinear", align_corners=False)
        l1   = (pred_img - target_img).abs().mean(dim=1, keepdim=True)
        ssim = self.ssim(pred_img, target_img).mean(dim=1, keepdim=True)
        return self.ssim_weight * ssim + (1.0 - self.ssim_weight) * l1

    def edge_aware_smoothness(
        self, depth: torch.Tensor, image: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourages depth to be smooth while preserving edges guided by RGB.
        Normalise depth to decouple scale from smoothness magnitude.
        Image is resized to match depth resolution if needed.
        """
        # Resize image to depth spatial size (depth is at P2 resolution, image may be full-res)
        if image.shape[2:] != depth.shape[2:]:
            image = F.interpolate(image, size=depth.shape[2:], mode="bilinear", align_corners=False)

        mean_depth = depth.mean(dim=(2, 3), keepdim=True)
        norm_depth = depth / (mean_depth + 1e-7)

        grad_depth_x = (norm_depth[:, :, :, :-1] - norm_depth[:, :, :, 1:]).abs()
        grad_depth_y = (norm_depth[:, :, :-1, :] - norm_depth[:, :, 1:, :]).abs()

        grad_img_x   = (image[:, :, :, :-1] - image[:, :, :, 1:]).abs().mean(1, keepdim=True)
        grad_img_y   = (image[:, :, :-1, :] - image[:, :, 1:, :]).abs().mean(1, keepdim=True)

        # Edge-aware weights: suppress smoothness where image edges are strong
        w_x = torch.exp(-grad_img_x)
        w_y = torch.exp(-grad_img_y)

        return (w_x * grad_depth_x).mean() + (w_y * grad_depth_y).mean()

    def forward(self, preds: dict, targets: dict) -> torch.Tensor:
        """
        Args:
            preds  : {"depth": (B,1,H,W), "warped_imgs": List[(B,3,H,W)]}
            targets: {"target_img": (B,3,H,W), "source_imgs": List[(B,3,H,W)]}
        """
        depth       = preds["depth"]
        target_img  = targets["target_img"]
        warped_imgs = preds.get("warped_imgs", [])

        # Align target_img to depth/warped resolution once (avoids repeated resizing)
        if warped_imgs and target_img.shape[2:] != warped_imgs[0].shape[2:]:
            target_img = F.interpolate(target_img, size=warped_imgs[0].shape[2:],
                                       mode="bilinear", align_corners=False)
        elif not warped_imgs and target_img.shape[2:] != depth.shape[2:]:
            target_img = F.interpolate(target_img, size=depth.shape[2:],
                                       mode="bilinear", align_corners=False)

        if not warped_imgs:
            # No warped images yet (e.g. first iteration): fall back to smoothness only
            return self.smooth_weight * self.edge_aware_smoothness(depth, target_img)

        # Per-pixel minimum photometric loss across source views (occlusion handling)
        photo_losses = [self.photometric_loss(w, target_img) for w in warped_imgs]
        photo_stack  = torch.cat(photo_losses, dim=1)   # (B, num_src, H, W)

        if self.auto_mask:
            # Identity warp baseline — resize source to match warped resolution
            src_imgs = targets.get("source_imgs", warped_imgs)
            identity_losses = []
            for src in src_imgs:
                if src.shape[2:] != target_img.shape[2:]:
                    src = F.interpolate(src, size=target_img.shape[2:],
                                        mode="bilinear", align_corners=False)
                identity_losses.append(self.photometric_loss(src, target_img))
            identity_stack = torch.cat(identity_losses, dim=1) + 1e-5
            photo_stack = torch.cat([photo_stack, identity_stack], dim=1)

        min_photo = photo_stack.min(dim=1, keepdim=True)[0]
        smooth    = self.edge_aware_smoothness(depth, target_img)

        return min_photo.mean() + self.smooth_weight * smooth