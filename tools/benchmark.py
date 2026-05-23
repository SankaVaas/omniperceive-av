"""
Latency & Throughput Benchmark — OmniPerceive
===============================================
Measures inference speed across three runtimes:
  1. PyTorch  (FP32 and FP16/AMP)
  2. ONNX Runtime (CPU and CUDA EP)
  3. TensorRT (if trt_path provided)

Usage:
    # PyTorch benchmark
    python tools/benchmark.py \
        --config configs/kitti_multitask.yaml \
        --checkpoint checkpoints/kitti/best.pth

    # ONNX Runtime benchmark
    python tools/benchmark.py \
        --onnx weights/omniperceive_kitti.onnx \
        --input_h 384 --input_w 1280

Outputs:
    - Mean / std latency (ms) per forward pass
    - Throughput (FPS)
    - GPU memory peak (MB) — CUDA only
    - Per-head breakdown (optional, --breakdown)

Colab/Kaggle T4 note:
    CUDA warmup is essential on T4 (first few runs are always slow).
    This script runs --warmup=20 iterations before timing.
"""

import argparse
import time
import torch
import numpy as np
import yaml
from pathlib import Path

from utils.logger import get_logger


def parse_args():
    p = argparse.ArgumentParser("OmniPerceive Benchmark")
    p.add_argument("--config",     default=None, help="YAML config (for PyTorch)")
    p.add_argument("--checkpoint", default=None, help="Checkpoint (for PyTorch)")
    p.add_argument("--onnx",       default=None, help="ONNX file path")
    p.add_argument("--input_h",    type=int, default=384)
    p.add_argument("--input_w",    type=int, default=1280)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup",     type=int, default=20)
    p.add_argument("--runs",       type=int, default=100)
    p.add_argument("--fp16",       action="store_true")
    p.add_argument("--breakdown",  action="store_true",
                   help="Time each head individually")
    return p.parse_args()


def benchmark_pytorch(args, logger):
    """Time PyTorch inference with CUDA events for sub-millisecond accuracy."""
    import yaml
    from models.omniperceive import OmniPerceive
    from utils.checkpoint    import load_checkpoint

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = OmniPerceive(cfg["model"]).to(device).eval()
    load_checkpoint(args.checkpoint, model, logger=logger)

    if args.fp16 and device.type == "cuda":
        model = model.half()

    dummy = torch.randn(args.batch_size, 3, args.input_h, args.input_w, device=device)
    if args.fp16 and device.type == "cuda":
        dummy = dummy.half()

    logger.info(f"PyTorch benchmark | device={device} | fp16={args.fp16}")
    logger.info(f"Input: {tuple(dummy.shape)}")

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    if device.type == "cuda":
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        mem_before = torch.cuda.memory_allocated(device)

        with torch.no_grad():
            for _ in range(args.runs):
                start_evt.record()
                _ = model(dummy)
                end_evt.record()
                torch.cuda.synchronize()
                latencies.append(start_evt.elapsed_time(end_evt))   # ms

        mem_peak = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        with torch.no_grad():
            for _ in range(args.runs):
                t0 = time.perf_counter()
                _ = model(dummy)
                latencies.append((time.perf_counter() - t0) * 1000)
        mem_peak = None

    _print_results(latencies, args.batch_size, mem_peak, "PyTorch", logger)

    # Per-head breakdown
    if args.breakdown:
        _benchmark_heads(model, dummy, device, args, logger)


def _benchmark_heads(model, dummy, device, args, logger):
    """Time each head in isolation using shared features."""
    logger.info("\nPer-head breakdown:")
    with torch.no_grad():
        fpn_feats, bev_feats = model.extract_features(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        for name, fn, feats in [
            ("DetectionHead",    model.det_head,   bev_feats),
            ("LaneHead",         model.lane_head,  bev_feats),
            ("DepthHead",        model.depth_head, fpn_feats),
            ("SegmentationHead", model.seg_head,   fpn_feats),
        ]:
            lats = []
            for _ in range(50):
                if device.type == "cuda":
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record(); fn(feats); e.record()
                    torch.cuda.synchronize()
                    lats.append(s.elapsed_time(e))
                else:
                    t0 = time.perf_counter(); fn(feats)
                    lats.append((time.perf_counter()-t0)*1000)
            logger.info(f"  {name:20s}  {np.mean(lats):.2f} ± {np.std(lats):.2f} ms")


def benchmark_onnx(args, logger):
    """Benchmark ONNX Runtime (CPU and CUDA EP if available)."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not installed. pip install onnxruntime-gpu")
        return

    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("ONNX Runtime: CUDA EP available ✅")
    else:
        providers = ["CPUExecutionProvider"]
        logger.info("ONNX Runtime: CPU EP only")

    sess = ort.InferenceSession(args.onnx, providers=providers)

    dummy  = np.random.randn(args.batch_size, 3, args.input_h, args.input_w).astype(np.float32)
    inputs = {"image": dummy}

    logger.info(f"ONNX benchmark | model={args.onnx}")

    # Warmup
    for _ in range(args.warmup):
        sess.run(None, inputs)

    # Timed
    latencies = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        sess.run(None, inputs)
        latencies.append((time.perf_counter() - t0) * 1000)

    _print_results(latencies, args.batch_size, None, "ONNX Runtime", logger)


def _print_results(latencies, batch_size, mem_peak_mb, tag, logger):
    lats = np.array(latencies)
    mean_ms  = lats.mean()
    std_ms   = lats.std()
    p50      = np.percentile(lats, 50)
    p95      = np.percentile(lats, 95)
    p99      = np.percentile(lats, 99)
    fps      = 1000.0 / mean_ms * batch_size

    logger.info(f"\n{'='*50}")
    logger.info(f"[{tag}] Results (batch={batch_size})")
    logger.info(f"  Mean latency : {mean_ms:.2f} ± {std_ms:.2f} ms")
    logger.info(f"  P50 / P95 / P99: {p50:.2f} / {p95:.2f} / {p99:.2f} ms")
    logger.info(f"  Throughput   : {fps:.1f} FPS")
    if mem_peak_mb:
        logger.info(f"  GPU memory   : {mem_peak_mb:.1f} MB")
    logger.info(f"{'='*50}\n")

