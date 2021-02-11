import torch.nn as nn


class MultiStepLoss(nn.Module):
    """
    Wrapper to wrap a loss function that can be applied
    on a prediction where each element of the last
    dimension should be predicting the same target (but where
    actual prediction and its quality might be different).

    """

    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = loss_function

    def forward(self, pred, target, steps=None):
        """
        """
        if steps:
            target = target.unsqueeze(1).repeat(1, steps)
        return self.loss_function(pred, target)
