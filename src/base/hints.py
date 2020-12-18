from typing import Union, Callable

import torch

# loss functions are torch.nn.Module that
# - in input: (predictions : Tensor[N,*], target : Tensor[N,*] )
# - in output: scalar if reduction is used otherwise Tensor[N,*]
Criterion = Union[torch.nn.Module, Callable[[torch.Tensor, torch.Tensor], Union[torch.Tensor, float]]]

# TODO: Allow more tensors as input
# (prediction,target) -> float
Metric = Union[torch.nn.Module, Callable[[torch.Tensor, torch.Tensor], float]]

__all__ = ["Criterion", "Metric"]
