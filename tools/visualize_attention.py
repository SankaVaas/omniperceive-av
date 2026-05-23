import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import cv2
import yaml

from models.omniperceive import OmniPerceive
from utils.checkpoint    import load_checkpoint
from utils.logger        import get_logger
from utils.visualization import apply_colormap, unnormalise_image


def parse_args():
    p = argparse.ArgumentParser("OmniPerceive Attention Visualiser")
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image",      required=True,   help="Path to input image (PNG/JPG)")
    p.add_argument("--output",     default="results/attention/")
    p.add_argument("--heads",      nargs="+", type=int, default=None,
                   help="Which attn heads to show (default: averaged)")
    p.add_argument("--stage",      type=int, default=None,
                   help="Which Swin stage to hook (0-3, default: all)")
    p.add_argument("--token_idx",  type=int, default=None,
                   help="Show attention FROM a specific token. Default: mean over all.")
    return p.parse_args()


def load_image(path: str, cfg: dict) -> tuple:
    """Load and preprocess a single image. Returns (tensor, bgr_original)."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    H = cfg["dataset"]["img_height"]
    W = cfg["dataset"]["img_width"]
    rgb_rs = cv2.resize(rgb, (W, H))

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (rgb_rs.astype(np.float32) / 255.0 - mean) / std   # (H,W,3)
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)

    bgr_rs = cv2.resize(bgr, (W, H))
    return tensor, bgr_rs


def extract_multi_stage_attention(
    model:     OmniPerceive,
    image:     torch.Tensor,
    stage_ids: list,
) -> dict:
    """
    Register forward hooks on multiple Swin stages and collect attention weights.

    Returns:
        dict: stage_id → attention tensor (num_heads, N_tokens, N_tokens)
    """
    attn_maps = {}
    hooks     = []

    def make_hook(stage_id):
        def hook_fn(module, inp, output):
            # Average over batch*windows dimension, keep heads
            attn_maps[stage_id] = attn.detach().mean(dim=0)  # (num_heads, N, N)
        return hook_fn

    # Hook the last transformer block of each requested stage
    if hasattr(model.backbone, "layers"):
        for sid in stage_ids:
            if sid < len(model.backbone.layers):
                block = model.backbone.layers[sid].blocks[-1]
                hooks.append(block.attn.register_forward_hook(make_hook(sid)))

    model.eval()
    with torch.no_grad():
        model(image)

    for h in hooks:
        h.remove()

    return attn_maps


def attn_to_image(
    attn:      torch.Tensor,    # (num_heads, N, N) or (N, N)
    feat_h:    int,
    feat_w:    int,
    img_h:     int,
    img_w:     int,
    head_idx:  int  = None,     # None = average all heads
    token_idx: int  = None,     # None = mean over all query tokens
) -> np.ndarray:
    """
    Convert raw attention weight tensor to a (img_h, img_w, 3) BGR heatmap.

    Strategy:
      - If head_idx given: use that head's attention
      - Else: average over all heads (more stable, less noisy)
      - If token_idx given: show attention FROM that token → others
      - Else: mean over all query tokens (global attention density)
    """
    attn = attn.cpu().float()

    if attn.dim() == 3:
        if head_idx is not None:
            a = attn[head_idx]     # (N, N)
        else:
            a = attn.mean(dim=0)   # (N, N)  — avg over heads
    else:
        a = attn   # (N, N)

    if token_idx is not None:
        a = a[token_idx]     # (N,) — attention from specific token
    else:
        a = a.mean(dim=0)    # (N,) — mean over query tokens

    N = a.shape[0]
    # Window-partitioned attention: N = window_size^2, reshape to window grid
    # Swin uses 7x7 windows; N tokens is per-window, not global
    # We recover a global attention map via reshape to (feat_h, feat_w)
    # (approximate: assumes global tokens, good enough for visualisation)
    s = int(N ** 0.5)
    if s * s != N:
        # Pad or trim to nearest square
        s = min(feat_h, feat_w, s)
    a_2d = a[:s * s].view(s, s).numpy()

    # Upsample to image resolution
    a_up = cv2.resize(a_2d, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    a_up = np.clip(a_up, 0, None)
    a_up = (a_up - a_up.min()) / (a_up.max() - a_up.min() + 1e-8)

    heatmap = apply_colormap(a_up, cmap=cv2.COLORMAP_HOT)
    return heatmap


def save_attention_panel(
    bgr_img:    np.ndarray,
    attn_maps:  dict,
    feat_sizes: dict,
    output_dir: Path,
    head_idx:   int = None,
    token_idx:  int = None,
):
    """
    Save a side-by-side panel: [RGB | Stage1 | Stage2 | Stage3 attn maps]
    overlaid on the image.
    """
    H, W = bgr_img.shape[:2]
    panels = [bgr_img]
    stage_labels = []

    for sid, attn in sorted(attn_maps.items()):
        fh, fw = feat_sizes.get(sid, (H // (2 ** (sid + 2)), W // (2 ** (sid + 2))))
        heatmap = attn_to_image(attn, fh, fw, H, W, head_idx, token_idx)
        overlay = cv2.addWeighted(bgr_img, 0.45, heatmap, 0.55, 0)

        # Label the panel
        cv2.putText(overlay, f"Stage {sid+1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        panels.append(overlay)
        stage_labels.append(sid)

        # Save individual stage map
        cv2.imwrite(str(output_dir / f"attention_stage{sid+1}.png"), heatmap)
        cv2.imwrite(str(output_dir / f"attention_overlay_stage{sid+1}.png"), overlay)

    # Combined panel: [original | stage1 | stage2 | stage3]
    panel_w = W * len(panels)
    panel   = np.concatenate(panels, axis=1)

    # Header bar
    header = np.zeros((40, panel_w, 3), dtype=np.uint8)
    titles = ["RGB Input"] + [f"Stage {s+1} Attention" for s in stage_labels]
    for i, title in enumerate(titles):
        cv2.putText(header, title, (W * i + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    full_panel = np.concatenate([header, panel], axis=0)
    panel_path = output_dir / "attention_panel.png"
    cv2.imwrite(str(panel_path), full_panel)
    return panel_path


def main():
    args   = parse_args()
    logger = get_logger("attn_vis", args.output)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")   # attention vis on CPU is fine for single image
    model  = OmniPerceive(cfg["model"]).to(device)
    load_checkpoint(args.checkpoint, model, logger=logger)
    model.eval()

    # Load image
    image_tensor, bgr_img = load_image(args.image, cfg)
    image_tensor = image_tensor.to(device)
    H, W = bgr_img.shape[:2]
    logger.info(f"Image loaded: {args.image}  ({H}×{W})")

    # Determine which stages to hook
    num_stages = len(cfg["model"]["backbone"].get("depths", [2,2,6,2]))
    stage_ids  = [args.stage] if args.stage is not None else list(range(num_stages))
    logger.info(f"Extracting attention from stages: {stage_ids}")

    # Extract
    attn_maps = extract_multi_stage_attention(model, image_tensor, stage_ids)
    logger.info(f"Got attention from {len(attn_maps)} stages: {list(attn_maps.keys())}")

    if not attn_maps:
        logger.warning(
            "No attention maps captured. The Swin WindowAttention in this build "
            "does not return attn weights by default.\n"
            "To enable: modify SwinWindowAttention.forward() to return "
            "(x, attn_weights) and set return_attn=True in SwinBlock."
        )
        sys.exit(0)

    # Feature sizes (H/4, H/8, H/16, H/32 for Swin stages 0–3)
    feat_sizes = {
        sid: (H // (4 * 2**sid), W // (4 * 2**sid))
        for sid in stage_ids
    }

    # Save output
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    head_idx = args.heads[0] if args.heads else None

    panel_path = save_attention_panel(
        bgr_img, attn_maps, feat_sizes, out_dir,
        head_idx=head_idx, token_idx=args.token_idx,
    )
    logger.info(f"Panel saved → {panel_path}")

    # Multi-head grid (if --heads specified)
    if args.heads and len(args.heads) > 1:
        for sid, attn in sorted(attn_maps.items()):
            fh, fw = feat_sizes[sid]
            rows = []
            for hid in args.heads:
                hm = attn_to_image(attn, fh, fw, H//2, W//2, head_idx=hid)
                rows.append(hm)
            grid = np.concatenate(rows, axis=1)
            cv2.imwrite(str(out_dir / f"attention_multihead_stage{sid+1}.png"), grid)
            logger.info(f"Multi-head grid saved for stage {sid+1}")

    logger.info("Done ✅")
    logger.info(f"All outputs in: {out_dir}")

