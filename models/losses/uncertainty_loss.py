"""
Homoscedastic Uncertainty-Weighted Multi-Task Loss
===================================================
Implements Kendall & Gal (2018) "Multi-Task Learning Using Uncertainty
to Weigh Losses for Scene Geometry and Semantics" — CVPR 2018.

Each task gets a learnable log-variance (log_sigma^2). The total loss is:

    L_total = sum_i [ (1 / (2 * sigma_i^2)) * L_i  +  log(sigma_i) ]

Key properties:
  - sigma_i is learned end-to-end (no manual weight tuning)
  - log(sigma_i) acts as a regularizer preventing sigma → ∞
  - Works for both regression (L2/smooth-L1) and classification (CE) tasks
"""

import torch
import torch.nn as nn
from typing import List


class HomoscedasticUncertaintyLoss(nn.Module):
    """
    Uncertainty-weighted combination of N task losses.

    Args:
        num_tasks (int): Number of tasks (one log_sigma per task).
        init_log_sigma (float): Initial value for log(sigma). Default -0.5
            gives sigma ≈ 0.61 — a reasonable starting weight ≈ 1.35×.

    Attributes:
        log_sigma (nn.Parameter): Shape (num_tasks,). Learnable per-task
            log uncertainty. Monitored in TensorBoard as 'log_sigma/task_i'.

    Example::
        criterion = HomoscedasticUncertaintyLoss(num_tasks=4)
        loss, log_vars = criterion([l_det, l_lane, l_depth, l_seg])
        loss.backward()
    """

    def __init__(self, num_tasks: int, init_log_sigma: float = -0.5):
        super().__init__()
        # Initialise as a free parameter — gradients flow through here
        self.log_sigma = nn.Parameter(
            torch.full((num_tasks,), init_log_sigma, dtype=torch.float32)
        )
        self.num_tasks = num_tasks

    def forward(self, task_losses: List[torch.Tensor]):
        """
        Args:
            task_losses: List of scalar tensors [L_det, L_lane, L_depth, L_seg].

        Returns:
            total_loss (torch.Tensor): Scalar, backward-able.
            log_vars   (torch.Tensor): (num_tasks,) log_sigma values for logging.
        """
        assert len(task_losses) == self.num_tasks, (
            f"Expected {self.num_tasks} losses, got {len(task_losses)}"
        )

        total = torch.zeros(1, device=self.log_sigma.device)
        for i, l_i in enumerate(task_losses):
            # precision = 1 / (2 * sigma^2) = exp(-log_sigma^2)
            precision = torch.exp(-2.0 * self.log_sigma[i])
            total = total + precision * l_i + self.log_sigma[i]

        return total.squeeze(), self.log_sigma.detach()

    def extra_repr(self) -> str:
        return (f"num_tasks={self.num_tasks}, "
                f"log_sigma init={self.log_sigma.data.tolist()}")
