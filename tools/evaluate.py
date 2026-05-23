"""
Evaluation Script — OmniPerceive
==================================
Runs full validation over a dataset split and reports all task metrics.

Usage:
    python tools/evaluate.py \
        --config  configs/kitti_multitask.yaml \
        --checkpoint checkpoints/kitti/best.pth \
        [--split val] [--vis] [--output results/]

Outputs (to --output dir):
    metrics.json        — all task metrics as JSON
    depth_errors.png    — depth error histogram
    det_pr_curves.png   — per-class precision-recall curves
    sample_vis/         — visualised predictions for first N samples

Colab tip:
    Mount Google Drive first, then pass --checkpoint from Drive path.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np
import yaml
from tqdm import tqdm

from models.omniperceive import OmniPerceive
from datasets import build_dataset
from datasets.transforms import build_transforms
from utils.checkpoint import load_checkpoint
from utils.logger import get_logger
from utils.metrics import (
    DepthMetricAggregator,
    SegMetricAggregator,
    DetectionEvaluator,
    compute_lane_metrics,
)
from utils.visualization import (
    vis_depth, vis_segmentation, vis_bev_detections, make_training_panel
)
from datasets.utils import box3d_to_bev


def parse_args():
    p = argparse.ArgumentParser("OmniPerceive Evaluation")
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split",      default="val")
    p.add_argument("--vis",        action="store_true",
                   help="Save visualised predictions for first 50 samples")
    p.add_argument("--vis_n",      type=int, default=50)
    p.add_argument("--output",     default="results/")
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


@torch.no_grad()
def evaluate(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("evaluate", args.output)
    logger.info(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = OmniPerceive(cfg["model"]).to(device).eval()
    load_checkpoint(args.checkpoint, model, logger=logger)

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = build_dataset(cfg, split=args.split)
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(2, os.cpu_count() or 1),
        collate_fn=getattr(dataset, "collate_fn", None),
    )
    logger.info(f"Evaluating on {len(dataset)} samples ({args.split} split)")

    # ── Metric aggregators ────────────────────────────────────────────────
    depth_agg = DepthMetricAggregator(
        min_d=cfg["model"]["heads"]["depth"].get("eval_min_depth", 1e-3),
        max_d=cfg["model"]["heads"]["depth"].get("eval_max_depth", 80.0),
        use_median_scaling=cfg["model"]["heads"]["depth"].get("eval_use_median_scaling", True),
    )
    seg_agg = SegMetricAggregator()
    det_eval = DetectionEvaluator(
        num_classes=cfg["model"]["heads"]["detection"]["num_classes"],
        class_names=cfg["dataset"]["class_names"],
        iou_thresh=cfg["evaluation"].get("iou_thresh_car", 0.5),
    )
    lane_tp = lane_fp = lane_fn = 0

    # ── Output dirs ───────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "sample_vis"
    if args.vis:
        vis_dir.mkdir(exist_ok=True)

    # ── Evaluation loop ───────────────────────────────────────────────────
    bev_cfg = cfg["model"]["bev_neck"]

    for i, batch in enumerate(tqdm(loader, desc="Eval")):
        images  = batch["image"].to(device)
        targets = batch["targets"]

        outputs = model(images)          # inference — no targets needed

        # ── Depth ─────────────────────────────────────────────────────────
        depth_pred = outputs["depth"]["depth"]           # (B,1,H_p2,W_p2)
        depth_gt   = batch.get("depth")
        if depth_gt is not None:
            for b in range(images.shape[0]):
                pred_np = depth_pred[b, 0].cpu().numpy()
                gt_np   = depth_gt[b].cpu().numpy() if depth_gt.dim() == 3 \
                          else depth_gt[b, 0].cpu().numpy()
                # Resize pred to GT resolution for eval
                import cv2
                if pred_np.shape != gt_np.shape:
                    pred_np = cv2.resize(pred_np, (gt_np.shape[1], gt_np.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)
                depth_agg.update(pred_np, gt_np)

        # ── Segmentation ──────────────────────────────────────────────────
        seg_logits = outputs["seg"]                      # (B,1,H_p2,W_p2)
        seg_gt     = targets.get("seg_mask")
        if seg_gt is not None:
            seg_pred_bin = (torch.sigmoid(seg_logits) > 0.5).cpu().numpy()
            seg_gt_bin   = (seg_gt > 0.5).cpu().numpy()
            for b in range(images.shape[0]):
                seg_agg.update(seg_pred_bin[b, 0], seg_gt_bin[b, 0])

        # ── Detection ─────────────────────────────────────────────────────
        det_out  = outputs["det"]
        for b in range(images.shape[0]):
            dets = model.det_head.decode(
                {k: v[b:b+1] for k, v in det_out.items()},
                score_thresh=cfg["model"]["heads"]["detection"].get("score_thresh", 0.25),
            )
            pred_boxes   = dets["bev_boxes"].numpy() if len(dets["bev_boxes"]) else np.zeros((0,4))
            pred_scores  = dets["scores"].numpy()    if len(dets["scores"])    else np.zeros(0)
            pred_classes = dets["classes"].numpy()   if len(dets["classes"])   else np.zeros(0, int)

            gt_boxes_3d  = targets["boxes_3d"][b]    if isinstance(targets["boxes_3d"], list) \
                           else targets["boxes_3d"][b].numpy()
            gt_classes   = targets["class_ids"][b]   if isinstance(targets["class_ids"], list) \
                           else targets["class_ids"][b].numpy()

            if len(gt_boxes_3d) > 0:
                gt_bev = box3d_to_bev(
                    np.array(gt_boxes_3d),
                    bev_cfg["pc_range"], bev_cfg["bev_h"], bev_cfg["bev_w"]
                )
            else:
                gt_bev = np.zeros((0, 4))

            det_eval.update(pred_boxes, pred_scores, pred_classes,
                            gt_bev, np.array(gt_classes))

        # ── Visualise ─────────────────────────────────────────────────────
        if args.vis and i < args.vis_n:
            for b in range(min(images.shape[0], 2)):
                panel = make_training_panel(
                    image=images[b].cpu(),
                    depth_pred=depth_pred[b].cpu(),
                    seg_pred=seg_logits[b].cpu(),
                    bev_heatmap=det_out["heatmap"][b].cpu(),
                    panel_h=256, panel_w=256,
                )
                import torchvision
                torchvision.utils.save_image(panel, vis_dir / f"sample_{i:04d}_{b}.png")

    # ── Aggregate + print ─────────────────────────────────────────────────
    results = {}

    depth_m = depth_agg.compute()
    seg_m   = seg_agg.compute()
    det_m   = det_eval.compute()

    results.update({f"depth/{k}": v for k, v in depth_m.items()})
    results.update({f"seg/{k}":   v for k, v in seg_m.items()})
    results.update({f"det/{k}":   v for k, v in det_m.items()})

    logger.info("=" * 60)
    logger.info("DEPTH METRICS")
    for k, v in depth_m.items():
        logger.info(f"  {k:12s}: {v:.4f}")

    logger.info("SEGMENTATION METRICS")
    for k, v in seg_m.items():
        logger.info(f"  {k:12s}: {v:.4f}")

    logger.info("DETECTION METRICS")
    for k, v in det_m.items():
        logger.info(f"  {k:12s}: {v:.4f}")

    logger.info("=" * 60)

    # Save JSON
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    return results


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)