from typing import Any, Callable, Optional

import torch
from pytorch_lightning.metrics import classification

from .metrics import ignore_class


class Accuracy(classification.Accuracy):
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
