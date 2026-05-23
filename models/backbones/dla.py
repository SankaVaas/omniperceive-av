"""
Deep Layer Aggregation (DLA) Backbone — DLA-34
================================================
DLA learns hierarchical feature aggregation through iterative deep
aggregation (IDA) and hierarchical deep aggregation (HDA) nodes.

Why DLA-34 for AV perception:
  - Lightweight (15M params) → runs on T4/V100 at full KITTI resolution
  - Designed for dense prediction (depth, segmentation) — no spatial info loss
  - Used in CenterNet, CenterPoint as backbone of choice
  - IDA upsampling preserves fine spatial detail for depth/lane heads
  - No ImageNet classification head — outputs are directly multi-scale maps

Architecture overview:
    Input (3, H, W)
      └─ Level 0: BasicBlock ×1  → (16, H,    W   )
      └─ Level 1: BasicBlock ×1  → (32, H/2,  W/2 )
      └─ Level 2: BasicBlock ×1  → (64, H/4,  W/4 )
      └─ Level 3: BasicBlock ×2  → (128,H/8,  W/8 )   ← out_indices[0]
      └─ Level 4: BasicBlock ×2  → (256,H/16, W/16)   ← out_indices[1]
      └─ Level 5: ResBlock ×1    → (512,H/32, W/32)   ← out_indices[2]
      └─ IDA upsampling merges 3→5 → same-resolution aggregated maps

Reference: Yu et al. "Deep Layer Aggregation" — CVPR 2018.
           https://arxiv.org/abs/1707.06484
CenterNet: Zhou et al. 2019 — https://arxiv.org/abs/1904.07850
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ── Basic Conv Block ────────────────────────────────────────────────────────

def conv_bn_relu(
    in_ch: int, out_ch: int,
    kernel: int = 3, stride: int = 1, padding: int = 1,
    dilation: int = 1, bias: bool = False,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                  padding=padding, dilation=dilation, bias=bias),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock: 3×3 Conv → BN → ReLU → 3×3 Conv → BN + skip.
    Used for DLA levels 0-4.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.skip(x))


class Bottleneck(nn.Module):
    """
    ResNet Bottleneck: 1×1 → 3×3 → 1×1. Used at level 5 for DLA-34.
    expansion=2 (DLA convention, unlike ResNet's 4).
    """
    expansion = 2

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        mid_ch = out_ch // self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + self.skip(x))


# ── IDA (Iterative Deep Aggregation) Node ──────────────────────────────────

class IDANode(nn.Module):
    """
    Iterative Deep Aggregation: upsample + project two adjacent feature levels
    and sum them. This is the fine-grained aggregation across resolutions.

    IDA refines shallow (high-res) features with deep (low-res) context without
    losing spatial precision — critical for depth and lane detection.

    Args:
        out_ch  : output channel count for both branches.
        in_chs  : [ch_shallow, ch_deep] input channels.
        scale   : upsampling factor for the deep branch (typically 2).
    """

    def __init__(self, out_ch: int, in_chs: List[int], scale: int = 2):
        super().__init__()
        # Project each input branch to out_ch
        self.proj_shallow = conv_bn_relu(in_chs[0], out_ch, kernel=1, padding=0)
        self.proj_deep    = conv_bn_relu(in_chs[1], out_ch, kernel=1, padding=0)

        # Learnable upsampling: deformable or bilinear init conv
        # We use a standard transposed conv here (lighter than deformable)
        self.up = nn.ConvTranspose2d(
            out_ch, out_ch,
            kernel_size=scale * 2, stride=scale,
            padding=scale // 2,
            groups=out_ch,   # depthwise — one filter per channel
            bias=False,
        )
        nn.init.constant_(self.up.weight, 1.0 / (scale ** 2))  # bilinear init

        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, shallow: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shallow : (B, C_s, H, W)    — high-resolution, shallow features
            deep    : (B, C_d, H/s, W/s) — low-resolution, deep features

        Returns:
            aggregated : (B, out_ch, H, W)
        """
        x_s = self.proj_shallow(shallow)
        x_d = self.up(self.proj_deep(deep))

        # Align sizes (may differ by 1px due to odd dimensions)
        if x_d.shape != x_s.shape:
            x_d = F.interpolate(x_d, size=x_s.shape[2:], mode="bilinear", align_corners=False)

        return self.relu(self.norm(x_s + x_d))


# ── HDA (Hierarchical Deep Aggregation) Root ───────────────────────────────

class HDARoot(nn.Module):
    """
    HDA aggregation root: merges features from N parallel branches (each
    at the same resolution) by concatenation + 1×1 projection.

    In DLA, this sits at the top of each tree, aggregating the two branches
    within a stage before feeding downstream.

    Args:
        in_chs : list of channel counts per branch to merge.
        out_ch : output channel count.
    """

    def __init__(self, in_chs: List[int], out_ch: int, kernel: int = 1):
        super().__init__()
        total_in = sum(in_chs)
        self.conv = nn.Conv2d(total_in, out_ch, kernel,
                              padding=kernel // 2, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(torch.cat(xs, dim=1))))


# ── DLA-34 Backbone ─────────────────────────────────────────────────────────

class DLA34(nn.Module):
    """
    DLA-34: Deep Layer Aggregation backbone for dense AV perception.

    Output channels at each level:
        Level 2: 64   (H/4)
        Level 3: 128  (H/8)   ← out_indices[0]
        Level 4: 256  (H/16)  ← out_indices[1]
        Level 5: 512  (H/32)  ← out_indices[2]

    With IDA upsampling, levels 3-5 are all returned at H/8 resolution
    and combined by the IDA aggregation chain. Without IDA (IDA disabled),
    raw multi-scale maps are returned for FPN input.

    Args:
        pretrained     : path to pretrained .pth or None.
        out_indices    : which levels to return (0-indexed from level 2).
        use_ida        : enable IDA upsampling aggregation.
        return_raw_maps: if True, return raw level maps (no IDA aggregation),
                         suited for FPN input.
    """

    # Channel counts per level [0..5]
    CHANNELS = [16, 32, 64, 128, 256, 512]

    def __init__(
        self,
        pretrained:      Optional[str] = None,
        out_indices:     List[int] = [1, 2, 3],   # relative to level 2
        use_ida:         bool = False,             # False → raw maps for FPN
        return_raw_maps: bool = True,
    ):
        super().__init__()
        C = self.CHANNELS
        self.out_indices     = out_indices
        self.use_ida         = use_ida
        self.return_raw_maps = return_raw_maps

        # ── Stem levels 0-2 (no downsampling at level 0-1) ────────────────
        self.level0 = self._make_level(BasicBlock,  3,    C[0], 1, stride=1)
        self.level1 = self._make_level(BasicBlock,  C[0], C[1], 1, stride=2)
        self.level2 = self._make_level(BasicBlock,  C[1], C[2], 1, stride=2)

        # ── Deep levels 3-5 (HDA tree structure) ──────────────────────────
        # Level 3: two branches of BasicBlock, merged by HDARoot
        self.level3_b0   = self._make_level(BasicBlock, C[2], C[3], 2, stride=2)
        self.level3_b1   = self._make_level(BasicBlock, C[2], C[3], 2, stride=2)
        
        self.level3_root = HDARoot([C[3], C[3]], C[3])

        # Level 4: two branches of BasicBlock
        self.level4_b0   = self._make_level(BasicBlock, C[3], C[4], 2, stride=2)
        self.level4_b1   = self._make_level(BasicBlock, C[3], C[4], 2, stride=2)
        
        self.level4_root = HDARoot([C[4], C[4]], C[4])

        # Level 5: Bottleneck
        self.level5 = self._make_level(Bottleneck, C[4], C[5], 1, stride=2)

        # IDA aggregation chain (levels 3→4→5 → fused at level-3 resolution)
        if use_ida:
            self.ida_3_4 = IDANode(C[3], [C[3], C[4]], scale=2)
            self.ida_4_5 = IDANode(C[3], [C[3], C[5]], scale=4)

        self._init_weights()

        if pretrained:
            self.load_pretrained(pretrained)

    @staticmethod
    def _make_level(
        block, in_ch: int, out_ch: int, num_blocks: int, stride: int = 1
    ) -> nn.Sequential:
        layers = [block(in_ch, out_ch, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(block(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def load_pretrained(self, path: str):
        """Load pretrained DLA-34 weights (e.g. from CenterPoint or DLA repo)."""
        import os
        if not os.path.exists(path):
            print(f"[DLA34] Pretrained not found at {path}. Training from scratch.")
            return
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        # Strip 'backbone.' prefix if loaded from full detector checkpoint
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=False)
        print(f"[DLA34] Loaded pretrained from {path} | "
              f"missing={len(missing)}, unexpected={len(unexpected)}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x : (B, 3, H, W)

        Returns (return_raw_maps=True, out_indices=[1,2,3]):
            [(B, 128, H/8, W/8),
             (B, 256, H/16,W/16),
             (B, 512, H/32,W/32)]

        Returns (use_ida=True):
            [(B, 128, H/8, W/8)] — single aggregated output (CenterNet mode)
        """
        # Stem
        x0 = self.level0(x)                        # (B, 16, H, W)
        x1 = self.level1(x0)                       # (B, 32, H/2, W/2)
        x2 = self.level2(x1)                       # (B, 64, H/4, W/4)

        # Level 3 HDA tree
        # Both branches stride-2 from x2 → same H/8 resolution
        l3_b0  = self.level3_b0(x2)                # (B, 128, H/8, W/8)
        l3_b1  = self.level3_b1(x2)                # (B, 128, H/8, W/8)
        # IDA: upsample l3_b0 (already H/8) — scale=1 passthrough; here we
        # just use it to project+fuse with a pooled x2 to inject shallow ctx
        # Simpler correct formulation: root merges the two branches directly
        x3 = self.level3_root(l3_b0, l3_b1)        # (B, 128, H/8, W/8)

        # Level 4 HDA tree
        l4_b0  = self.level4_b0(x3)                # (B, 256, H/16, W/16)
        l4_b1  = self.level4_b1(x3)                # (B, 256, H/16, W/16)
        x4 = self.level4_root(l4_b0, l4_b1)        # (B, 256, H/16, W/16)

        # Level 5
        x5 = self.level5(x4)                       # (B, 512, H/32, W/32)

        if self.use_ida:
            # IDA aggregation: pull everything to H/8 resolution
            agg_4 = self.ida_3_4(x3, x4)           # (B, 128, H/8, W/8)
            agg_5 = self.ida_4_5(agg_4, x5)        # (B, 128, H/8, W/8)
            return [agg_5]

        # Raw multi-scale maps for FPN
        raw = [x2, x3, x4, x5]                     # levels 2,3,4,5
        return [raw[i] for i in self.out_indices]


def build_dla(cfg: dict) -> DLA34:
    """Instantiate DLA34 from config dict."""
    return DLA34(
        pretrained=cfg.get("pretrained"),
        out_indices=cfg.get("out_indices", [1, 2, 3]),
        use_ida=cfg.get("use_ida", False),
        return_raw_maps=cfg.get("return_raw_maps", True),
    )