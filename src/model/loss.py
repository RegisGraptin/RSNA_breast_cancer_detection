
from typing import Optional
import torch

from torch import Tensor

from torch.nn.modules.loss import _Loss, _WeightedLoss
import torch.nn.functional as F

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight = None, reduction = 'mean', smoothing = 0.0, pos_weight = None):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing  = smoothing
        self.weight     = weight
        self.reduction  = reduction
        self.pos_weight = pos_weight

    @staticmethod
    def _smooth(targets, n_labels, smoothing = 0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad(): targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1), self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight, pos_weight = self.pos_weight)
        if  self.reduction == 'sum': loss = loss.sum()
        elif  self.reduction == 'mean': loss = loss.mean()
        return loss
    

class FocalLoss:
    """Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002."""
    
    def __init__(self, gamma: float, alpha, reduction: str = 'sum') -> None:
        """

        Note:
        
            alpha: float = -1,
            gamma: float = 2,

        Args:
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.

            weight/alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction


    def forward(self, inputs, targets):
        inputs  = inputs.float()
        targets = targets.float()

        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
