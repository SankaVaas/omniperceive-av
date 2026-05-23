"""
Visualisation Utilities
========================
TensorBoard-ready image panels for all four tasks:
  - Depth colourmap (error map, pred vs GT)
  - BEV detection overlay (predicted + GT boxes)
  - Segmentation overlay on RGB
  - Lane overlay on BEV canvas
  - Per-task attention maps from backbone
  - Training dashboard panel (all tasks in one grid)

All functions return (C, H, W) float32 tensors in [0, 1]
suitable for TensorBoard's add_image() / add_images().
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Optional


# ── Colourmap helpers ──────────────────────────────────────────────────────

def apply_colormap(
    gray:    np.ndarray,   # (H, W) float, will be normalised to [0,1]
    cmap:    int = cv2.COLORMAP_MAGMA,
    vmin:    Optional[float] = None,
    vmax:    Optional[float] = None,
) -> np.ndarray:
    """Apply an OpenCV colourmap to a greyscale array. Returns (H, W, 3) uint8 BGR."""
    if vmin is None: vmin = gray[gray > 0].min() if (gray > 0).any() else 0
    if vmax is None: vmax = gray.max()
    norm = np.clip((gray - vmin) / (vmax - vmin + 1e-8), 0, 1)
    gray_u8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray_u8, cmap)   # (H, W, 3) BGR


def bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 BGR → (3, H, W) float32 [0,1] RGB."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def unnormalise_image(
    img_tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std:  List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """(3,H,W) normalised tensor → (H,W,3) uint8 BGR for OpenCV."""
    img = img_tensor.cpu().float()
    for c, (m, s) in enumerate(zip(mean, std)):
        img[c] = img[c] * s + m
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ── Depth visualisation ────────────────────────────────────────────────────

def vis_depth(
    pred_depth: torch.Tensor,    # (1, H, W) or (H, W)
    gt_depth:   Optional[torch.Tensor] = None,   # same shape, 0=invalid
    max_depth:  float = 80.0,
) -> torch.Tensor:
    """
    Returns a (3, H, W*2) panel: [pred_colourmap | error_map].
    If gt_depth is None, returns just the pred colourmap (3, H, W).
    """
    pred = pred_depth.squeeze().cpu().numpy()
    pred_col = apply_colormap(pred, cmap=cv2.COLORMAP_MAGMA, vmin=0, vmax=max_depth)

    if gt_depth is None:
        return bgr_to_tensor(pred_col)

    gt = gt_depth.squeeze().cpu().numpy()
    valid = gt > 0
    err   = np.zeros_like(pred)
    if valid.any():
        err[valid] = np.abs(pred[valid] - gt[valid]) / (gt[valid] + 1e-8)
    err_col = apply_colormap(err, cmap=cv2.COLORMAP_JET, vmin=0, vmax=0.3)

    panel = np.concatenate([pred_col, err_col], axis=1)   # side by side
    return bgr_to_tensor(panel)


# ── Segmentation visualisation ─────────────────────────────────────────────

def vis_segmentation(
    image:    torch.Tensor,      # (3, H, W) normalised RGB
    seg_pred: torch.Tensor,      # (1, H, W) logits or probs
    seg_gt:   Optional[torch.Tensor] = None,   # (1, H, W) binary
    alpha:    float = 0.5,
) -> torch.Tensor:
    """
    Overlay predicted segmentation on RGB image.
    Returns (3, H, W*2) panel: [pred overlay | GT overlay] or just pred.
    """
    bgr = unnormalise_image(image)                         # (H,W,3) uint8 BGR
    H, W = bgr.shape[:2]

    pred_mask = (torch.sigmoid(seg_pred).squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    def overlay(img, mask, color=(0, 200, 0)):
        canvas = img.copy()
        canvas[mask == 1] = (
            (1 - alpha) * canvas[mask == 1] + alpha * np.array(color)
        ).astype(np.uint8)
        return canvas

    pred_vis = overlay(bgr, pred_mask)

    if seg_gt is None:
        return bgr_to_tensor(pred_vis)

    gt_mask = seg_gt.squeeze().cpu().numpy().astype(np.uint8)
    gt_vis  = overlay(bgr, gt_mask, color=(0, 0, 200))

    panel = np.concatenate([pred_vis, gt_vis], axis=1)
    return bgr_to_tensor(panel)


# ── BEV detection visualisation ───────────────────────────────────────────

def vis_bev_detections(
    bev_h:     int,
    bev_w:     int,
    pred_boxes:   np.ndarray,     # (N, 4) [cx,cy,w,h] BEV pixels
    pred_scores:  np.ndarray,     # (N,)
    pred_classes: np.ndarray,     # (N,)
    gt_boxes:     Optional[np.ndarray] = None,   # (M, 4)
    gt_classes:   Optional[np.ndarray] = None,
    score_thresh: float = 0.3,
) -> torch.Tensor:
    """
    Returns (3, bev_h, bev_w*2) if GT provided else (3, bev_h, bev_w).
    Predicted boxes in colour, GT in white dashed boxes.
    """
    from utils.bev_utils import render_bev_detections, render_bev_gt

    pred_canvas = render_bev_detections(
        bev_h, bev_w, pred_boxes, pred_classes, pred_scores,
        score_thresh=score_thresh
    )

    if gt_boxes is None or len(gt_boxes) == 0:
        return bgr_to_tensor(pred_canvas)

    gt_canvas = render_bev_gt(bev_h, bev_w, gt_boxes, gt_classes)
    panel = np.concatenate([pred_canvas, gt_canvas], axis=1)
    return bgr_to_tensor(panel)


# ── Attention map visualisation ────────────────────────────────────────────

def vis_attention_map(
    image:      torch.Tensor,          # (3, H, W) normalised
    attn_map:   torch.Tensor,          # (num_heads, N, N) or (1, H_a, W_a)
    head_idx:   int = 0,
    upsample:   bool = True,
) -> torch.Tensor:
    """
    Overlay backbone self-attention map on input image.

    Handles both Swin-T attention (num_heads, N_tokens, N_tokens) and
    pre-averaged (1, H_a, W_a) formats.

    Returns (3, H, W) tensor.
    """
    bgr = unnormalise_image(image)
    H, W = bgr.shape[:2]

    attn = attn_map.cpu().float()

    if attn.dim() == 3 and attn.shape[0] > 1:
        # (num_heads, N, N) — average over keys for one head
        attn = attn[head_idx].mean(dim=0)     # (N,)
        n    = attn.shape[0]
        s    = int(n ** 0.5)
        attn = attn[:s * s].view(1, 1, s, s)
    elif attn.dim() == 3:
        attn = attn.unsqueeze(0)              # (1,1,H_a,W_a)
    elif attn.dim() == 1:
        s = int(attn.shape[0] ** 0.5)
        attn = attn[:s*s].view(1, 1, s, s)

    if upsample:
        attn = torch.nn.functional.interpolate(
            attn, size=(H, W), mode="bilinear", align_corners=False
        )

    attn_np = attn.squeeze().numpy()
    attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
    attn_col = apply_colormap(attn_np, cmap=cv2.COLORMAP_HOT)

    blend = cv2.addWeighted(bgr, 0.5, attn_col, 0.5, 0)
    return bgr_to_tensor(blend)


# ── Combined training dashboard ────────────────────────────────────────────

def make_training_panel(
    image:       torch.Tensor,              # (3, H, W)
    depth_pred:  torch.Tensor,             # (1, h, w)
    seg_pred:    torch.Tensor,             # (1, h, w)
    bev_heatmap: Optional[torch.Tensor],  # (C, bev_h, bev_w)
    depth_gt:    Optional[torch.Tensor] = None,
    seg_gt:      Optional[torch.Tensor] = None,
    panel_h:     int = 256,
    panel_w:     int = 256,
) -> torch.Tensor:
    """
    Assemble a 2×2 grid panel for TensorBoard:
        [RGB image    | Depth colourmap]
        [Seg overlay  | Det heatmap    ]

    All panels are resized to (panel_h, panel_w) for uniform layout.
    Returns (3, panel_h*2, panel_w*2).
    """
    def to_bgr_resized(tensor_chw, cmap=None):
        """Convert any (1 or 3, H, W) tensor to resized BGR uint8."""
        t = tensor_chw.cpu().float()
        if t.shape[0] == 1:
            arr = t.squeeze().numpy()
            if cmap:
                bgr = apply_colormap(arr, cmap=cmap)
            else:
                arr = np.clip((arr - arr.min())/(arr.max()-arr.min()+1e-8),0,1)
                bgr = (arr[...,None] * 255).repeat(3,2).astype(np.uint8)
        else:
            bgr = unnormalise_image(t)
        return cv2.resize(bgr, (panel_w, panel_h))

    # Top-left: RGB
    tl = to_bgr_resized(image)

    # Top-right: depth (magma colourmap)
    tr = to_bgr_resized(depth_pred, cmap=cv2.COLORMAP_MAGMA)

    # Bottom-left: segmentation overlay
    seg_np  = torch.sigmoid(seg_pred).squeeze().cpu().numpy()
    seg_mask = (seg_np > 0.5).astype(np.uint8)
    img_bgr = cv2.resize(unnormalise_image(image), (panel_w, panel_h))
    img_bgr[cv2.resize(seg_mask,(panel_w,panel_h)).astype(bool)] = \
        (img_bgr[cv2.resize(seg_mask,(panel_w,panel_h)).astype(bool)] * 0.5 +
         np.array([0,200,0]) * 0.5).astype(np.uint8)
    bl = img_bgr

    # Bottom-right: BEV heatmap (max over classes)
    if bev_heatmap is not None:
        hm = bev_heatmap.max(dim=0).values.cpu().numpy()
        br = apply_colormap(hm, cmap=cv2.COLORMAP_HOT, vmin=0, vmax=1)
        br = cv2.resize(br, (panel_w, panel_h))
    else:
        br = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    top    = np.concatenate([tl, tr], axis=1)    # (H, 2W, 3)
    bottom = np.concatenate([bl, br], axis=1)    # (H, 2W, 3)
    panel  = np.concatenate([top, bottom], axis=0)  # (2H, 2W, 3)

    return bgr_to_tensor(panel)