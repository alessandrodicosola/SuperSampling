from typing import Union, Callable

import torch

Criterion = torch.nn.modules.loss._Loss

# TODO: Allow more tensors as input
# (prediction,target) -> float
Metric = Union[torch.nn.Module, Callable[[torch.Tensor, torch.Tensor], float]]

__all__ = ["Criterion", "Metric"]
