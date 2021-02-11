from typing import Any, Callable, Optional

import torch
from pytorch_lightning.metrics import classification, Metric
from pytorch_lightning.metrics.utils import _input_format_classification
from .metrics import ignore_class, BatchSplitter


class Accuracy(classification.Accuracy):
    """
    Accuracy, with ignore_index to be able to
    ignore all samples of a given class.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        ignore_index: int = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
            threshold=threshold,
        )

        self.ignore_index = ignore_index

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        return super().update(
            *ignore_class(preds, target, ignore_index=self.ignore_index)
        )


class FractionSolved(Metric):
    """
    Fraction of subgraphs for which all nodes were predicted
    correctly.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        ignore_index: int = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.ignore_index = ignore_index
        self.threshold = threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor, batch: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """

        _target = target
        preds, target = ignore_class(preds, target, ignore_index=self.ignore_index)
        batch, _ = ignore_class(batch, _target, ignore_index=self.ignore_index)
        preds, target = _input_format_classification(preds, target, self.threshold)
        assert preds.shape == target.shape

        splitter = BatchSplitter(batch)
        correct = 0
        for subset in splitter(preds == target):
            correct += subset.all()

        self.correct += correct
        self.total += splitter.n_graphs

    def compute(self):
        """
        Computes fraction of solved graphs over state.
        """
        return self.correct.float() / self.total