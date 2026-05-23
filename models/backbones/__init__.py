"""
Backbone registry.
Resolves cfg["backbone"]["type"] → backbone class.
"""

from models.backbones.swin_transformer import SwinTransformer, build_swin
from models.backbones.dla              import DLA34, build_dla

_REGISTRY = {
    "SwinTransformer": build_swin,
    "DLA34":           build_dla,
}


def build_backbone(cfg: dict):
    """
    Build backbone from config dict.

    Args:
        cfg : backbone section of YAML config, must contain 'type' key.

    Example cfg:
        type: SwinTransformer
        pretrained: weights/swin_tiny_patch4_window7_224.pth
        embed_dim: 96
        depths: [2, 2, 6, 2]
        ...
    """
    btype = cfg.get("type")
    if btype not in _REGISTRY:
        raise KeyError(
            f"Unknown backbone '{btype}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[btype](cfg)


__all__ = ["SwinTransformer", "DLA34", "build_backbone"]