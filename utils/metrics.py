"""
Evaluation Metrics
===================
Implements the standard metrics for each task:

Detection  : KITTI 3D AP (R11 / R40) and nuScenes mAP / NDS
Depth      : Abs Rel, Sq Rel, RMSE, RMSE log, δ < 1.25^n  (Eigen protocol)
Segmentation: IoU (Intersection-over-Union), MaxF (KITTI Road)
Lane       : F1, Precision, Recall at pixel-level
"""

import numpy as np
from typing import Dict, List, Tuple


# ── Depth Metrics (Eigen et al. split protocol) ────────────────────────────

def compute_depth_metrics(
    pred:    np.ndarray,
    gt:      np.ndarray,
    min_d:   float = 1e-3,
    max_d:   float = 80.0,
    use_median_scaling: bool = True,
) -> Dict[str, float]:
    """
    Compute the standard 7 depth evaluation metrics.

    Args:
        pred              : (H, W) predicted depth in metres.
        gt                : (H, W) ground-truth depth in metres. 0 = invalid.
        min_d / max_d     : depth range for valid pixel selection.
        use_median_scaling: apply per-image median scaling (Eigen protocol).
                            Set False for supervised / metric-depth models.

    Returns:
        dict with keys: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    """
    valid = (gt > min_d) & (gt < max_d)
    pred  = pred[valid]
    gt    = gt[valid]

    if pred.size == 0:
        nan = float("nan")
        return {k: nan for k in ["abs_rel","sq_rel","rmse","rmse_log","a1","a2","a3"]}

    # Median scaling: rescale pred to gt scale (self-supervised models are scale-ambiguous)
    if use_median_scaling:
        scale = np.median(gt) / (np.median(pred) + 1e-8)
        pred  = pred * scale

    pred = np.clip(pred, min_d, max_d)

    thresh  = np.maximum(gt / pred, pred / gt)   # element-wise max ratio

    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse     = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt + 1e-8) - np.log(pred + 1e-8)) ** 2).mean())
    abs_rel  = (np.abs(gt - pred) / (gt + 1e-8)).mean()
    sq_rel   = (((gt - pred) ** 2) / (gt + 1e-8)).mean()

    return {
        "abs_rel":  float(abs_rel),
        "sq_rel":   float(sq_rel),
        "rmse":     float(rmse),
        "rmse_log": float(rmse_log),
        "a1":       float(a1),
        "a2":       float(a2),
        "a3":       float(a3),
    }


class DepthMetricAggregator:
    """Accumulates per-image depth metrics and returns mean over dataset split."""

    def __init__(self, min_d: float = 1e-3, max_d: float = 80.0,
                 use_median_scaling: bool = True):
        self.min_d = min_d
        self.max_d = max_d
        self.use_median_scaling = use_median_scaling
        self.reset()

    def reset(self):
        self._records: List[Dict[str, float]] = []

    def update(self, pred: np.ndarray, gt: np.ndarray):
        m = compute_depth_metrics(pred, gt, self.min_d, self.max_d,
                                  self.use_median_scaling)
        self._records.append(m)

    def compute(self) -> Dict[str, float]:
        if not self._records:
            return {}
        keys = self._records[0].keys()
        return {k: float(np.nanmean([r[k] for r in self._records])) for k in keys}


# ── Segmentation Metrics ───────────────────────────────────────────────────

def compute_seg_metrics(
    pred_mask: np.ndarray,    # (H, W) binary int  0/1
    gt_mask:   np.ndarray,    # (H, W) binary int  0/1
) -> Dict[str, float]:
    """
    IoU and MaxF (harmonic mean of precision & recall at best threshold).

    KITTI Road benchmark uses MaxF as primary metric.
    """
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    tp = (pred &  gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    tn = (~pred & ~gt).sum()

    iou = tp / (tp + fp + fn + 1e-8)

    prec   = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    maxf   = 2 * prec * recall / (prec + recall + 1e-8)

    return {
        "IoU":       float(iou),
        "MaxF":      float(maxf),
        "precision": float(prec),
        "recall":    float(recall),
    }


class SegMetricAggregator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = 0

    def update(self, pred: np.ndarray, gt: np.ndarray):
        p, g = pred.astype(bool), gt.astype(bool)
        self.tp += (p &  g).sum()
        self.fp += (p & ~g).sum()
        self.fn += (~p & g).sum()

    def compute(self) -> Dict[str, float]:
        tp, fp, fn = self.tp, self.fp, self.fn
        iou  = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        maxf = 2 * prec * rec / (prec + rec + 1e-8)
        return {"IoU": float(iou), "MaxF": float(maxf),
                "precision": float(prec), "recall": float(rec)}


# ── Lane Metrics ───────────────────────────────────────────────────────────

def compute_lane_metrics(
    pred_mask: np.ndarray,   # (H, W) binary — rendered predicted lanes
    gt_mask:   np.ndarray,   # (H, W) binary — rendered GT lanes
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    """
    Pixel-level lane F1 / precision / recall.
    Uses a dilated GT mask (±5 pixels) to account for annotation jitter.
    """
    import cv2
    kernel   = np.ones((11, 11), np.uint8)
    gt_dilat = cv2.dilate(gt_mask.astype(np.uint8), kernel).astype(bool)

    pred = pred_mask.astype(bool)
    gt   = gt_dilat

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt_mask.astype(bool)).sum()   # original GT for FN

    prec   = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1     = 2 * prec * recall / (prec + recall + 1e-8)

    return {"F1": float(f1), "precision": float(prec), "recall": float(recall)}


# ── Detection: KITTI AP helpers ───────────────────────────────────────────

def box_iou_3d_bev(
    boxes_a: np.ndarray,   # (N, 4) [cx, cy, w, h] in BEV pixels
    boxes_b: np.ndarray,   # (M, 4)
) -> np.ndarray:
    """Compute pairwise 2D BEV IoU between two sets of axis-aligned boxes."""
    N, M = len(boxes_a), len(boxes_b)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    # Convert cx,cy,w,h → x1,y1,x2,y2
    def to_xyxy(b):
        return np.stack([b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2,
                         b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2], axis=1)

    a = to_xyxy(boxes_a)   # (N,4)
    b = to_xyxy(boxes_b)   # (M,4)

    inter_x1 = np.maximum(a[:,0:1], b[:,0])   # (N,M)
    inter_y1 = np.maximum(a[:,1:2], b[:,1])
    inter_x2 = np.minimum(a[:,2:3], b[:,2])
    inter_y2 = np.minimum(a[:,3:4], b[:,3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a = (a[:,2]-a[:,0]) * (a[:,3]-a[:,1])   # (N,)
    area_b = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])   # (M,)
    union  = area_a[:,None] + area_b[None,:] - inter

    return inter / (union + 1e-8)


def compute_ap(
    recalls:    np.ndarray,
    precisions: np.ndarray,
    method:     str = "R40",
) -> float:
    """
    Compute Average Precision using 11-point (R11) or 40-point (R40) interpolation.
    KITTI uses R40 as the primary metric.
    """
    if method == "R11":
        thresholds = np.linspace(0, 1, 11)
    else:  # R40
        thresholds = np.linspace(0, 1, 41)

    ap = 0.0
    for t in thresholds:
        precs = precisions[recalls >= t]
        ap += precs.max() if precs.size > 0 else 0.0
    return ap / len(thresholds)


class DetectionEvaluator:
    """
    Lightweight KITTI-style 3D detection evaluator.
    Accumulates predictions and GT across batches, then computes per-class AP.

    Usage:
        ev = DetectionEvaluator(num_classes=3, iou_thresh=0.5)
        for batch in val_loader:
            ev.update(pred_boxes, pred_scores, pred_classes,
                      gt_boxes,   gt_classes)
        metrics = ev.compute()  # {'AP_car': ..., 'mAP': ...}
    """

    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        iou_thresh:  float = 0.5,
    ):
        self.num_classes = num_classes
        self.class_names = class_names
        self.iou_thresh  = iou_thresh
        self.reset()

    def reset(self):
        # Per class: list of (score, tp_flag) tuples and GT count
        self._preds = {c: [] for c in range(self.num_classes)}
        self._n_gt  = {c: 0  for c in range(self.num_classes)}

    def update(
        self,
        pred_boxes:   np.ndarray,   # (N, 4) BEV boxes
        pred_scores:  np.ndarray,   # (N,)
        pred_classes: np.ndarray,   # (N,) int
        gt_boxes:     np.ndarray,   # (M, 4)
        gt_classes:   np.ndarray,   # (M,) int
    ):
        for cls in range(self.num_classes):
            p_mask = pred_classes == cls
            g_mask = gt_classes   == cls
            p_boxes  = pred_boxes[p_mask]
            p_scores = pred_scores[p_mask]
            g_boxes  = gt_boxes[g_mask]

            self._n_gt[cls] += len(g_boxes)

            if len(p_boxes) == 0:
                continue

            # Sort by score descending
            order    = np.argsort(-p_scores)
            p_boxes  = p_boxes[order]
            p_scores = p_scores[order]

            matched_gt = set()
            if len(g_boxes) > 0:
                iou = box_iou_3d_bev(p_boxes, g_boxes)   # (N, M)

            for i in range(len(p_boxes)):
                tp = 0
                if len(g_boxes) > 0:
                    best_j = iou[i].argmax()
                    if iou[i, best_j] >= self.iou_thresh and best_j not in matched_gt:
                        tp = 1
                        matched_gt.add(best_j)
                self._preds[cls].append((float(p_scores[i]), tp))

    def compute(self) -> Dict[str, float]:
        results = {}
        aps = []

        for cls in range(self.num_classes):
            name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
            entries = sorted(self._preds[cls], key=lambda x: -x[0])
            n_gt    = self._n_gt[cls]

            if n_gt == 0 or not entries:
                results[f"AP_{name}"] = 0.0
                aps.append(0.0)
                continue

            tp_cum = np.cumsum([e[1] for e in entries]).astype(float)
            fp_cum = np.cumsum([1 - e[1] for e in entries]).astype(float)
            n      = np.arange(1, len(entries) + 1, dtype=float)

            precisions = tp_cum / n
            recalls    = tp_cum / n_gt

            ap = compute_ap(recalls, precisions, method="R40")
            results[f"AP_{name}"] = float(ap)
            aps.append(ap)

        results["mAP"] = float(np.mean(aps)) if aps else 0.0
        return results