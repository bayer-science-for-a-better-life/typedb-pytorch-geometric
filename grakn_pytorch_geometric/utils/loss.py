from typing import Callable
import torch
import torch.nn as nn


class MultiStepLoss(nn.Module):
    """
    Wrapper to wrap a loss function that can be applied
    on a prediction where each element of the last
    dimension should be predicting the same target (but where
    actual prediction and its quality might be different).

    Args:
        loss_function (callable): loss function to be wrapped.

    """

    def __init__(self, loss_function: Callable):
        super().__init__()
        self.loss_function = loss_function

    def forward(self, pred: torch.Tensor, target: torch.Tensor, steps=None):
        """

        Args:
            pred (torch.tensor):
            target:
            steps:

        Returns:

        """
        if steps:
            target = target.unsqueeze(1).repeat(1, steps)
        return self.loss_function(pred, target)
