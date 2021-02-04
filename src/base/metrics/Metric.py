from functools import partial
from abc import ABC, abstractmethod

import torch


class Metric(torch.nn.Module, ABC):
    def __init__(self, reduction=None):
        super(Metric, self).__init__()
        assert reduction in [None, 'mean', 'sum'], "Invalid reduction type. Correct values: [None, mean, sum]."
        self.reduction = reduction
        assert self.reduction in ['mean', 'sum'], "Trainer support only float values. Correct values: [mean, sum]."

    @abstractmethod
    def compute_values_per_batch(self, *args: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, *args: torch.Tensor):
        with torch.no_grad():
            metric_in_batch: torch.Tensor = self.compute_values_per_batch(*args)

        if not self.reduction:
            return metric_in_batch
        elif self.reduction == "mean":
            return metric_in_batch.mean(dim=0).item()
        elif self.reduction == "sum":
            return metric_in_batch.sum(dim=0).item()
        else:
            # it's never called since the check is done in the init.
            raise RuntimeError("")
