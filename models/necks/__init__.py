"""
Neck registry.
Resolves cfg["neck"]["type"] → neck class.
"""

from models.necks.fpn      import FPN,     build_fpn
from models.necks.bev_neck import BEVNeck, build_bev_neck

_REGISTRY = {
    "FPN":     build_fpn,
    "BEVNeck": build_bev_neck,
}


def build_neck(cfg: dict):
    """
    Build a neck from config dict.

    Args:
        cfg : neck section of YAML config, must contain 'type' key.

    Example cfg (FPN):
        type: FPN
        in_channels: [192, 384, 768]
        out_channels: 256
        num_outs: 4

    Example cfg (BEVNeck):
        type: BEVNeck
        in_channels: 256
        bev_h: 128
        bev_w: 128
        pc_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    """
    ntype = cfg.get("type")
    if ntype not in _REGISTRY:
        raise KeyError(
            f"Unknown neck '{ntype}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[ntype](cfg)


__all__ = ["FPN", "BEVNeck", "build_neck"]