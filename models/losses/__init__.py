from models.losses.uncertainty_loss import HomoscedasticUncertaintyLoss
from models.losses.depth_loss       import SelfSupervisedDepthLoss, SSIM
from models.losses.multitask_loss   import MultiTaskUncertaintyLoss

__all__ = [
    "HomoscedasticUncertaintyLoss",
    "SelfSupervisedDepthLoss",
    "SSIM",
    "MultiTaskUncertaintyLoss",
]