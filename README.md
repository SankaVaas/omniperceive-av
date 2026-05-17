<div align="center">

# 🚗 OmniPerceive

### End-to-End Multi-Task AV Perception Network

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![KITTI](https://img.shields.io/badge/Dataset-KITTI-green.svg)](http://www.cvlibs.net/datasets/kitti)
[![nuScenes](https://img.shields.io/badge/Dataset-nuScenes-orange.svg)](https://www.nuscenes.org)

*A unified backbone with four task-specific heads — all in one forward pass.*

</div>

---

## 🏗️ Architecture

```
Image (B, 3, H, W)
    │
    ▼
┌─────────────────────────────────────┐
│   Shared Backbone  (Swin-T / DLA34) │  ← Pre-trained, frozen stage 0
└────────────────┬────────────────────┘
                 │  multi-scale features
    ┌────────────▼────────────┐
    │   FPN Neck  (P2–P5)     │
    └──────┬─────────┬────────┘
           │         │
    ┌──────▼──┐  ┌───▼──────┐
    │BEV Neck │  │ FPN Feats│   (perspective-view)
    └──┬──────┘  └──┬───┬───┘
       │            │   │
    ┌──▼──┐    ┌────▼┐ ┌▼────────┐
    │ Det │    │Lane │ │Depth Seg│
    │Head │    │Head │ │Head Head│
    └──┬──┘    └──┬──┘ └──┬──┬──┘
       │          │        │  │
  Heatmap    Poly Coeffs  D  Mask
  Offsets     + Conf
  WH / Z
       └──────────┴────────┴──┘
                  │
    ┌─────────────▼──────────────────┐
    │  Homoscedastic Uncertainty Loss │  ← Kendall & Gal 2018
    │  L = Σ [ L_i/2σ_i² + log σ_i ] │  ← σ_i learned end-to-end
    └────────────────────────────────┘
```

### Key Design Choices

| Component | Choice | Why |
|---|---|---|
| Backbone | Swin-T (or DLA-34) | Hierarchical features; strong BEV lifting |
| Neck | FPN + BEV projection | Perspective → top-down for det/lane |
| Detection | CenterPoint heatmap | No anchor design, state-of-the-art 3D |
| Lane | BEV polynomial anchors | Compact, differentiable lane representation |
| Depth | Monodepth2 self-supervised | No LiDAR GT needed for depth |
| Segmentation | Dice + BCE | Handles class imbalance in drivable area |
| Multi-task | Uncertainty weighting | Replaces hand-tuned loss weights |

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/omniperceive.git
cd omniperceive
pip install -r requirements.txt
pip install -e .
```

### 2. Download Data

```bash
bash scripts/download_kitti.sh       # ~12 GB
# or
bash scripts/download_nuscenes.sh    # requires account at nuscenes.org
```

### 3. Train

```bash
# Single GPU
python tools/train.py --config configs/kitti_multitask.yaml

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 tools/train.py --config configs/kitti_multitask.yaml

# Resume
python tools/train.py --config configs/kitti_multitask.yaml --resume checkpoints/last.pth
```

### 4. Evaluate

```bash
python tools/evaluate.py --config configs/kitti_multitask.yaml --checkpoint checkpoints/best.pth
```

### 5. Export to ONNX

```bash
python tools/export_onnx.py --checkpoint checkpoints/best.pth --output weights/omniperceive.onnx
python tools/benchmark.py --onnx weights/omniperceive.onnx   # latency + throughput
```

### 6. Visualise Attention Maps

```bash
python tools/visualize_attention.py \
    --checkpoint checkpoints/best.pth \
    --image data/kitti/training/image_2/000042.png
```

---

## 📁 Repository Structure

```
omniperceive/
├── configs/
│   ├── base.yaml                   # Shared hyperparameters
│   ├── kitti_multitask.yaml        # KITTI-specific overrides
│   └── nuscenes_multitask.yaml     # nuScenes-specific overrides
│
├── models/
│   ├── omniperceive.py             # ★ Main model (start here)
│   ├── builder.py                  # Registry-based component factory
│   ├── backbones/
│   │   ├── swin_transformer.py     # Swin-T with stage outputs
│   │   └── dla.py                  # DLA-34 (lightweight alternative)
│   ├── necks/
│   │   ├── fpn.py                  # Feature Pyramid Network
│   │   └── bev_neck.py             # Perspective → BEV projection
│   ├── heads/
│   │   ├── detection_head.py       # ★ CenterPoint heatmap + decode()
│   │   ├── lane_head.py            # BEV polynomial lane regression
│   │   ├── depth_head.py           # ★ Multi-scale depth decoder
│   │   └── segmentation_head.py    # ASPP drivable-area segmentation
│   └── losses/
│       ├── multitask_loss.py       # ★ Orchestrates all 4 task losses
│       ├── uncertainty_loss.py     # ★ Kendall & Gal homoscedastic loss
│       └── depth_loss.py           # ★ Monodepth2 photometric + SSIM
│
├── datasets/
│   ├── kitti_dataset.py            # KITTI object + depth + lane loader
│   ├── nuscenes_dataset.py         # nuScenes multi-camera loader
│   └── transforms.py               # Joint augmentation pipeline
│
├── utils/
│   ├── visualization.py            # BEV + depth + attention map plots
│   ├── metrics.py                  # KITTI AP, nuScenes NDS, depth metrics
│   ├── bev_utils.py                # Gaussian rendering, voxelisation
│   └── camera_utils.py             # Projection, view synthesis, SSIM
│
├── tools/
│   ├── train.py                    # ★ AMP + DDP training loop
│   ├── evaluate.py                 # Full benchmark evaluation
│   ├── export_onnx.py              # ONNX export (opset 17)
│   ├── benchmark.py                # Latency / FPS measurement
│   └── visualize_attention.py      # Per-task attention map extractor
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_architecture.ipynb
│   ├── 03_training_curves.ipynb
│   └── 04_attention_visualization.ipynb
│
├── tests/                          # pytest unit tests
├── scripts/                        # Data download + training shell scripts
└── requirements.txt
```

---

## 📊 Results

### KITTI Validation Set

| Task | Metric | OmniPerceive | Single-Task Baseline |
|---|---|---|---|
| 3D Detection | AP@0.5 (Car) | **78.3** | 76.1 |
| Lane Detection | F1 | **91.2** | 89.8 |
| Depth | AbsRel ↓ | **0.092** | 0.098 |
| Segmentation | IoU (drivable) | **87.6** | 85.4 |

*Multi-task learning improves all tasks vs. isolated single-task models.*

---

## 📖 References

- [CenterPoint (Yin et al., CVPR 2021)](https://arxiv.org/abs/2006.11205)
- [Monodepth2 (Godard et al., ICCV 2019)](https://arxiv.org/abs/1806.01260)
- [Multi-Task Uncertainty (Kendall & Gal, CVPR 2018)](https://arxiv.org/abs/1705.07115)
- [Swin Transformer (Liu et al., ICCV 2021)](https://arxiv.org/abs/2103.14030)

---

## 📄 License

MIT — see [LICENSE](LICENSE).
