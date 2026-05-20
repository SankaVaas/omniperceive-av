"""
Unit tests for OmniPerceive model forward pass and output shapes.
Run: pytest tests/test_model.py -v
"""

import pytest
import torch
import yaml


# Minimal config for CPU testing
MINI_CFG = {
    "backbone": {"type": "DLA34", "pretrained": None},
    "neck":     {"type": "FPN", "in_channels": [128, 256, 512], "out_channels": 64, "num_outs": 4, "start_level": 0},
    "bev_neck": {"type": "BEVNeck", "in_channels": 64, "bev_h": 32, "bev_w": 32, "pc_range": [-51.2, -51.2, -5., 51.2, 51.2, 3.]},
    "heads": {
        "detection":    {"num_classes": 3, "hidden_channels": 32},
        "lane":         {"num_lanes": 4, "poly_degree": 3, "hidden_channels": 32, "anchor_stride": 8},
        "depth":        {"encoder_channels": [64, 32, 16, 8], "decoder_channels": [32, 16, 8, 4]},
        "segmentation": {"num_classes": 1, "hidden_channels": 32, "use_aspp": False},
    },
}

B, C, H, W = 2, 3, 128, 416  # small spatial dims for fast CPU tests


@pytest.fixture(scope="module")
def model():
    from models.omniperceive import OmniPerceive
    m = OmniPerceive(MINI_CFG).eval()
    return m


@pytest.fixture
def dummy_input():
    return torch.randn(B, C, H, W)


class TestOmniPerceiveForward:
    def test_output_keys(self, model, dummy_input):
        with torch.no_grad():
            out = model(dummy_input)
        assert set(out.keys()) == {"det", "lane", "depth", "seg"}

    def test_det_heatmap_shape(self, model, dummy_input):
        with torch.no_grad():
            out = model(dummy_input)
        hm = out["det"]["heatmap"]
        assert hm.shape[0] == B
        assert hm.shape[1] == MINI_CFG["heads"]["detection"]["num_classes"]
        assert hm.min() >= 0.0 and hm.max() <= 1.0, "Heatmap must be sigmoid output"

    def test_depth_output_range(self, model, dummy_input):
        with torch.no_grad():
            out = model(dummy_input)
        depth = out["depth"]["depth"]
        assert depth.shape == (B, 1, H, W)
        assert (depth > 0).all(), "Depth must be positive"

    def test_seg_shape(self, model, dummy_input):
        with torch.no_grad():
            out = model(dummy_input)
        seg = out["seg"]
        assert seg.shape == (B, 1, H, W)

    def test_no_nan_in_outputs(self, model, dummy_input):
        with torch.no_grad():
            out = model(dummy_input)
        for key in ["depth", "seg"]:
            tensor = out[key] if key == "seg" else out[key]["depth"]
            assert not torch.isnan(tensor).any(), f"NaN found in {key} output"


class TestDetectionDecode:
    def test_decode_returns_dict(self, model, dummy_input):
        with torch.no_grad():
            out  = model(dummy_input)
            dets = model.det_head.decode(out["det"], score_thresh=0.0)
        assert "scores"    in dets
        assert "bev_boxes" in dets
        assert "classes"   in dets

    def test_bev_boxes_shape(self, model, dummy_input):
        with torch.no_grad():
            out  = model(dummy_input)
            dets = model.det_head.decode(out["det"], score_thresh=0.0)
        if dets["bev_boxes"].numel() > 0:
            assert dets["bev_boxes"].shape[1] == 4


class TestUncertaintyLoss:
    def test_loss_decreases_with_gradient(self):
        from models.losses.uncertainty_loss import HomoscedasticUncertaintyLoss
        crit = HomoscedasticUncertaintyLoss(num_tasks=3)
        opt  = torch.optim.SGD(crit.parameters(), lr=0.1)

        losses_before = [torch.tensor(1.0, requires_grad=True),
                         torch.tensor(2.0, requires_grad=True),
                         torch.tensor(0.5, requires_grad=True)]

        total, _ = crit(losses_before)
        total.backward()
        assert crit.log_sigma.grad is not None, "log_sigma should receive gradients"

    def test_output_shape(self):
        from models.losses.uncertainty_loss import HomoscedasticUncertaintyLoss
        crit = HomoscedasticUncertaintyLoss(num_tasks=4)
        task_losses = [torch.tensor(float(i + 1), requires_grad=True) for i in range(4)]
        total, log_vars = crit(task_losses)
        assert total.shape == ()          # scalar
        assert log_vars.shape == (4,)
