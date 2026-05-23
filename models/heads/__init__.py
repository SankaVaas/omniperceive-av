"""
Head registry — imports all four task heads.
"""

from models.heads.detection_head    import DetectionHead
from models.heads.lane_head         import LaneHead
from models.heads.depth_head        import DepthHead
from models.heads.segmentation_head import SegmentationHead

__all__ = ["DetectionHead", "LaneHead", "DepthHead", "SegmentationHead"]