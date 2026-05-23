"""
Drivable-Area Segmentation Head
=================================
Produces a per-pixel binary (or multi-class) segmentation mask from FPN
perspective-view features. Designed for drivable area and lane divider
segmentation as required by KITTI Road and nuScenes HD map benchmarks.

Architecture:
  FPN features (multi-scale)
      ↓  ASPP module (multi-scale context with atrous convolutions)
      ↓  Decoder with skip connections from FPN P2
      ↓  1×1 output conv → (B, num_classes, H, W)

Key design choices:
  - ASPP (Atrous Spatial Pyramid Pooling) from DeepLabV3 captures both
    local detail and large context (critical for road segmentation which
    spans the entire lower image).
  - Decoder adds P2 skip connection to recover fine-grained boundaries.
  - Output is at full FPN P2 resolution (H/8 × W/8), then upsampled
    to input resolution during loss computation and visualisation.
  - Dice + BCE combo loss (configured in multitask_loss.py) handles the
    extreme foreground/background class imbalance in road segmentation.

Reference:
  Chen et al. "Encoder-Decoder with Atrous Separable Convolution for
  Semantic Image Segmentation" (DeepLabV3+) ECCV 2018.
  https://arxiv.org/abs/1802.02611
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3,
                 stride: int = 1, padding: int = 1,
                 dilation: int = 1, bias: bool = False):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=stride,
                      padding=padding * dilation if k > 1 else 0,
                      dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (DeepLabV3).

    Applies parallel dilated convolutions at multiple rates to capture
    multi-scale context without losing resolution:
      - 1×1 conv           (rate=1)  → local features
      - 3×3 dilated conv   (rate=6)  → ~7×7 effective RF
      - 3×3 dilated conv   (rate=12) → ~25×25 effective RF
      - 3×3 dilated conv   (rate=18) → ~37×37 effective RF
      - Global average pool          → image-level context

    All branches are projected to out_channels then concatenated + fused.

    Args:
        in_channels  : input feature channels.
        out_channels : output channels per ASPP branch (fused = 5×out_channels → projected).
        dilations    : list of dilation rates (default: [1, 6, 12, 18]).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int = 256,
        dilations:    List[int] = [1, 6, 12, 18],
    ):
        super().__init__()

        self.branches = nn.ModuleList()

        # 1×1 conv (dilation=1, no padding needed)
        self.branches.append(ConvBnRelu(in_channels, out_channels, k=1, padding=0))

        # Dilated 3×3 convs
        for rate in dilations[1:]:
            self.branches.append(
                ConvBnRelu(in_channels, out_channels, k=3, dilation=rate)
            )

        # Global average pooling branch (image-level context)
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion: (num_branches + 1) × out_channels → out_channels
        num_branches  = len(dilations) + 1   # dilated branches + GAP
        self.project  = nn.Sequential(
            nn.Conv2d(num_branches * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)
        Returns:
            (B, out_channels, H, W) — same spatial size as input
        """
        H, W = x.shape[2:]

        branch_outs = [b(x) for b in self.branches]

        # GAP: pool to 1×1, then upsample back to (H, W)
        gap = self.gap_branch(x)
        gap = F.interpolate(gap, size=(H, W), mode="bilinear", align_corners=False)
        branch_outs.append(gap)

        out = torch.cat(branch_outs, dim=1)
        return self.project(out)


class SegDecoder(nn.Module):
    """
    Lightweight decoder with a skip connection from FPN P2.

    After ASPP on a coarser FPN level (P4), upsample and fuse with
    P2 to recover fine boundary detail. Matches DeepLabV3+ decoder.

    Args:
        aspp_channels   : output channels of ASPP.
        skip_channels   : channels of the P2 skip connection.
        decoder_channels: intermediate decoder channels.
    """

    def __init__(
        self,
        aspp_channels:    int,
        skip_channels:    int,
        decoder_channels: int = 128,
    ):
        super().__init__()

        # Project skip (P2) to fewer channels before fusion
        self.skip_proj = ConvBnRelu(skip_channels, 48, k=1, padding=0)

        # Fuse: upsampled ASPP + projected skip → decoder output
        self.fuse = nn.Sequential(
            ConvBnRelu(aspp_channels + 48, decoder_channels),
            ConvBnRelu(decoder_channels,   decoder_channels),
        )

    def forward(
        self,
        aspp_feat: torch.Tensor,   # (B, aspp_ch, H_low, W_low)
        skip_feat: torch.Tensor,   # (B, skip_ch, H_high, W_high) — FPN P2
    ) -> torch.Tensor:
        """Returns (B, decoder_channels, H_high, W_high)."""
        # Upsample ASPP feature to P2 resolution
        aspp_up = F.interpolate(
            aspp_feat, size=skip_feat.shape[2:],
            mode="bilinear", align_corners=False
        )
        skip    = self.skip_proj(skip_feat)
        fused   = torch.cat([aspp_up, skip], dim=1)
        return self.fuse(fused)


class SegmentationHead(nn.Module):
    """
    Full drivable-area segmentation head.

    Args:
        cfg         : head config dict with keys:
                        num_classes       (int)   : 1 (KITTI binary) or 2 (nuScenes)
                        hidden_channels   (int)   : decoder channels (default 128)
                        use_aspp          (bool)  : enable ASPP (default True)
                        aspp_dilations    (list)  : dilation rates for ASPP
        in_channels : FPN feature channels (all levels same, = out_channels of FPN).

    Forward input:
        fpn_feats : [P2, P3, P4, P5] — fine to coarse.
                    ASPP runs on P4 (good balance of context vs resolution).
                    Decoder uses P2 skip for boundary recovery.

    Forward output:
        dict:
          seg_logits : (B, num_classes, H_p2, W_p2) — raw logits (no sigmoid/softmax)
          seg_probs  : (B, num_classes, H_p2, W_p2) — sigmoid/softmax probabilities
    """

    def __init__(self, cfg: dict, in_channels: int):
        super().__init__()
        self.num_classes    = cfg.get("num_classes",     1)
        hidden              = cfg.get("hidden_channels", 128)
        use_aspp            = cfg.get("use_aspp",        True)
        aspp_dilations      = cfg.get("aspp_dilations",  [1, 6, 12, 18])
        aspp_out_ch         = 256

        # ── Context module ────────────────────────────────────────────────
        if use_aspp:
            # ASPP runs on P4 (stride 16) for large receptive field
            self.context = ASPPModule(
                in_channels=in_channels,
                out_channels=aspp_out_ch // 4,   # 64 per branch, 5×64=320 → projected to 256
                dilations=aspp_dilations,
            )
            context_out_ch = aspp_out_ch // 4    # after ASPP project
            # Re-project to consistent channel count
            self.context_proj = ConvBnRelu(context_out_ch, aspp_out_ch, k=1, padding=0)
        else:
            # Lightweight fallback: simple conv
            self.context = nn.Sequential(
                ConvBnRelu(in_channels, aspp_out_ch),
                ConvBnRelu(aspp_out_ch, aspp_out_ch),
            )
            self.context_proj = nn.Identity()

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder = SegDecoder(
            aspp_channels=aspp_out_ch,
            skip_channels=in_channels,   # P2 has same channel count (FPN normalises)
            decoder_channels=hidden,
        )

        # ── Output head ───────────────────────────────────────────────────
        self.output_conv = nn.Conv2d(hidden, self.num_classes, 1)
        nn.init.constant_(self.output_conv.bias, -4.6)   # init: predict background

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.output_conv:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, fpn_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            fpn_feats : [P2(fine), P3, P4, P5(coarse)] from FPN.
                        All have shape (B, in_channels, H_i, W_i).

        Returns:
            seg_logits : (B, num_classes, H_p2, W_p2)
                         Raw logits — sigmoid/softmax applied in loss or postprocess.

        Note on output resolution:
            Output is at P2 resolution (H/8 × W/8) relative to the full image.
            The loss function upsamples GT masks to match; during inference,
            upsample seg_logits to full image resolution for visualisation.
        """
        p2 = fpn_feats[0]   # finest: (B, C, H/8,  W/8)  — used as skip
        p4 = fpn_feats[2]   # coarser: (B, C, H/32, W/32) — used for context

        # Apply ASPP / context on P4 (large receptive field, reasonable resolution)
        ctx = self.context(p4)
        ctx = self.context_proj(ctx)

        # Decoder: upsample ctx to P2 resolution + fuse with P2 skip
        dec = self.decoder(ctx, p2)

        # Final output conv
        logits = self.output_conv(dec)   # (B, num_classes, H_p2, W_p2)

        return logits

    @torch.no_grad()
    def predict(
        self,
        fpn_feats:    List[torch.Tensor],
        target_size:  tuple = None,
        thresh:       float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Full inference: forward + sigmoid/softmax + optional upsample + threshold.

        Args:
            fpn_feats   : FPN features.
            target_size : (H, W) to upsample output to. None = stay at P2 resolution.
            thresh      : binary threshold for num_classes=1.

        Returns:
            dict:
              logits : raw output (B, num_classes, H, W)
              probs  : probabilities
              mask   : binary/argmax mask (B, H, W)
        """
        logits = self.forward(fpn_feats)

        if target_size is not None:
            logits = F.interpolate(
                logits, size=target_size,
                mode="bilinear", align_corners=False
            )

        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
            mask  = (probs.squeeze(1) > thresh).long()
        else:
            probs = torch.softmax(logits, dim=1)
            mask  = probs.argmax(dim=1)

        return {"logits": logits, "probs": probs, "mask": mask}