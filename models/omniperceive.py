"""
OmniPerceive: End-to-End Multi-Task AV Perception Network
==========================================================
Unified backbone with four task-specific heads:
  1. 3D Object Detection  (CenterPoint-style heatmap)
  2. Lane Detection       (BEV polynomial anchors)
  3. Monocular Depth      (self-supervised Monodepth2 loss)
  4. Drivable-Area Segmentation

Trained with Kendall & Gal homoscedastic uncertainty-weighted multi-task loss.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from models.backbones import build_backbone
from models.necks import build_neck
from models.heads import DetectionHead, LaneHead, DepthHead, SegmentationHead
from models.losses import MultiTaskUncertaintyLoss


class OmniPerceive(nn.Module):
    """
    Single-forward-pass multi-task perception model for autonomous driving.

    Architecture:
        Image → Backbone (Swin-T | DLA-34)
              → FPN Neck (multi-scale features)
              → BEV Neck (perspective → bird's-eye-view projection)
              ├── Detection Head    → heatmap + offset + wh (CenterPoint)
              ├── Lane Head         → BEV polynomial coefficients + conf
              ├── Depth Head        → per-pixel depth map (encoder-decoder)
              └── Segmentation Head → drivable-area binary mask

    Args:
        cfg (dict): Full model config loaded from YAML.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        # ── Shared Encoder ────────────────────────────────────────────────
        self.backbone = build_backbone(cfg["backbone"])
        self.neck     = build_neck(cfg["neck"])       # FPN
        self.bev_neck = build_neck(cfg["bev_neck"])   # BEV projection

        feat_channels = cfg["neck"]["out_channels"]

        # ── Task Heads ────────────────────────────────────────────────────
        self.det_head   = DetectionHead(cfg["heads"]["detection"],    feat_channels)
        self.lane_head  = LaneHead(cfg["heads"]["lane"],              feat_channels)
        self.depth_head = DepthHead(cfg["heads"]["depth"],            feat_channels)
        self.seg_head   = SegmentationHead(cfg["heads"]["segmentation"], feat_channels)

        # ── Learnable log-variance per task (Kendall & Gal 2018) ──────────
        self.criterion = MultiTaskUncertaintyLoss(num_tasks=4)

        self._init_weights()

    def _init_weights(self):
        for head in [self.det_head, self.lane_head, self.seg_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def extract_features(self, images: torch.Tensor):
        """Shared feature extraction: backbone + FPN + BEV neck."""
        x        = self.backbone(images)
        fpn_feats = self.neck(x)
        bev_feats = self.bev_neck(fpn_feats)
        return fpn_feats, bev_feats

    def forward(self, images: torch.Tensor, targets: Optional[Dict] = None) -> Dict:
        """
        Args:
            images  : (B, 3, H, W)
            targets : dict of GT tensors — only needed during training
        Returns:
            Training : loss dict  |  Inference : prediction dict
        """
        fpn_feats, bev_feats = self.extract_features(images)

        det_preds   = self.det_head(bev_feats)    # CenterPoint heatmap
        lane_preds  = self.lane_head(bev_feats)   # BEV poly coefficients
        depth_preds = self.depth_head(fpn_feats)  # dense depth map
        seg_preds   = self.seg_head(fpn_feats)    # drivable-area mask

        outputs = {"det": det_preds, "lane": lane_preds,
                   "depth": depth_preds, "seg": seg_preds}

        if self.training and targets is not None:
            return self.criterion(outputs, targets)

        return outputs

    def get_attention_maps(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Hook into the last Swin-T transformer block and extract attention
        weights for per-task interpretability visualization.
        """
        attention_maps = {}
        hooks = []

        def _make_hook(name):
            def hook(module, inp, out):
                if isinstance(out, tuple) and len(out) == 2:
                    attention_maps[name] = out[1].detach()
            return hook

        if hasattr(self.backbone, "layers"):
            last_block = self.backbone.layers[-1].blocks[-1]
            hooks.append(last_block.attn.register_forward_hook(_make_hook("backbone_last")))

        with torch.no_grad():
            self.forward(images)

        for h in hooks:
            h.remove()

        return attention_maps

    def export_onnx(self, save_path: str, input_shape: Tuple = (1, 3, 384, 1280)):
        """Export to ONNX (opset 17) for TensorRT / deployment benchmarking."""
        self.eval()
        dummy = torch.randn(*input_shape)
        torch.onnx.export(
            self, dummy, save_path, opset_version=17,
            input_names=["image"],
            output_names=["det_heatmap", "lane_coeffs", "depth_map", "seg_mask"],
            dynamic_axes={"image": {0: "batch_size"}},
        )
        print(f"[OmniPerceive] ONNX exported → {save_path}")
