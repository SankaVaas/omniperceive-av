"""
Self-Supervised Monocular Depth Head
======================================
Encoder-decoder with skip connections.
Outputs multi-scale depth maps used for Monodepth2 loss.

Design choices:
  - Uses FPN features as encoder (shares backbone with other tasks)
  - Decoder with bilinear upsampling + conv (avoids checkerboard artifacts)
  - Sigmoid output → disparity → depth via D = 1 / (a * disp + b)
  - Multi-scale outputs [1/1, 1/2, 1/4, 1/8] for loss at each scale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),    # ELU preferred for depth (no dead neurons)
        )


class DepthDecoder(nn.Module):
    """
    Progressive upsampling decoder with lateral skip connections from FPN.

    At each stage i:
      1. upconv  : projects the current feature from its channel count → dec_ch[i]
      2. cat     : concatenate with skip from enc_ch[i+1] (next finer FPN level)
      3. iconv   : fuse concatenated features → dec_ch[i]

    encoder_channels: [P5_ch, P4_ch, P3_ch, P2_ch] (coarse → fine, same as FPN out_ch)
    decoder_channels: [dec0,  dec1,  dec2,  dec3 ]
    """

    def __init__(self, encoder_channels: List[int], decoder_channels: List[int]):
        super().__init__()
        assert len(encoder_channels) == len(decoder_channels)
        N = len(encoder_channels)

        self.upconvs = nn.ModuleList()
        self.iconvs  = nn.ModuleList()

        for i in range(N):
            # Input to upconv:
            #   stage 0 → encoder_channels[0] (P5)
            #   stage i → decoder_channels[i-1] (previous decoder output)
            in_ch_up = encoder_channels[0] if i == 0 else decoder_channels[i - 1]
            self.upconvs.append(ConvBnRelu(in_ch_up, decoder_channels[i]))

            # Input to iconv: dec_ch[i] (from upconv) + enc_ch[i+1] (skip, if exists)
            skip_ch  = encoder_channels[i + 1] if i + 1 < N else 0
            self.iconvs.append(ConvBnRelu(decoder_channels[i] + skip_ch, decoder_channels[i]))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [P5, P4, P3, P2] — coarse to fine (reversed FPN list).
        Returns:
            outputs : list of decoder feature maps, coarse → fine.
        """
        x = features[0]   # P5 — coarsest
        outputs = []
        for i, (upconv, iconv) in enumerate(zip(self.upconvs, self.iconvs)):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = upconv(x)
            if i + 1 < len(features):
                skip = features[i + 1]
                # Align spatial size (in case of rounding mismatches)
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = iconv(x)
            outputs.append(x)
        return outputs   # [coarse, ..., fine]


class DepthHead(nn.Module):
    """
    Depth estimation head using FPN features.

    Args:
        cfg          : head config dict.
        in_channels  : channel count of the finest FPN feature (P2 typically).

    Returns (forward):
        {
          "depth"        : (B, 1, H, W)         — full-resolution depth
          "disp_scales"  : List[(B,1,h,w)]      — multi-scale disparities
          "warped_imgs"  : List[(B,3,H,W)]      — view-synthesised frames (if pose given)
        }
    """

    # Depth range clipping (meters)
    MIN_DEPTH = 0.1
    MAX_DEPTH = 100.0

    def __init__(self, cfg: dict, in_channels: int):
        super().__init__()
        dec_chs = cfg.get("decoder_channels", [128, 64, 32, 16])
        num_levels = len(dec_chs)
        # FPN normalises all levels to the same channel count (in_channels)
        enc_chs = [in_channels] * num_levels

        self.decoder = DepthDecoder(enc_chs, dec_chs)

        # Disparity output at each scale
        self.disp_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 1, 3, padding=1),
                nn.Sigmoid()         # output in (0,1) → converted to depth
            )
            for ch in dec_chs
        ])

        # Learned scale/shift for disp→depth: depth = 1 / (a*disp + b)
        self.register_buffer("a", torch.tensor(self.MAX_DEPTH - self.MIN_DEPTH))
        self.register_buffer("b", torch.tensor(self.MIN_DEPTH))

    def disp_to_depth(self, disp: torch.Tensor) -> torch.Tensor:
        """Convert sigmoid disparity to metric depth (approximate)."""
        return 1.0 / (self.a * disp + self.b)

    def forward(self, fpn_feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # fpn_feats: [P2(fine), P3, P4, P5(coarse)]
        # Decoder expects coarse→fine
        dec_feats = self.decoder(list(reversed(fpn_feats)))

        disp_scales = [head(f) for f, head in zip(dec_feats, self.disp_heads)]

        # Full-resolution depth (finest scale)
        finest_disp = disp_scales[-1]
        if finest_disp.shape[-2:] != fpn_feats[0].shape[-2:]:
            finest_disp = F.interpolate(
                finest_disp, size=fpn_feats[0].shape[-2:],
                mode="bilinear", align_corners=False
            )

        depth = self.disp_to_depth(finest_disp)

        return {
            "depth":       depth,
            "disp_scales": disp_scales,
            "warped_imgs": [],    # populated by pose+warp module during training
        }