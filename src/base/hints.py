"""
Define type hints for objects
"""
from pathlib import Path

from typing import List, Tuple, Mapping, Union, NoReturn, Callable, Dict, NewType

import torch

# TYPES


Optimizer_ = Union[torch.optim.Optimizer, List[torch.optim.Optimizer], Tuple[torch.optim.Optimizer, ...], Mapping[
    str, torch.optim.Optimizer]]

# loss functions are torch.nn.Module that
# - in input: (predictions : Tensor[N,*], target : Tensor[N,*] )
# - in output: scalar if reduction is used otherwise Tensor[N,*]
Criterion = Union[torch.nn.Module, Callable[[torch.Tensor, torch.Tensor], Union[torch.Tensor, float]]]

# TODO: Allow more tensors as input
# (prediction,target) -> float
Metric = Union[torch.nn.Module, Callable[[torch.Tensor, torch.Tensor], float]]

__all__ = ["List", "Tuple", "Mapping", "Union", "NoReturn"]

__all__ += ["Tensor", "Optimizer_", "Criterion", "Metric"]
