"""
Define type hints for objects
"""
from pathlib import Path
from typing import List, Tuple, Mapping, Union, NoReturn, Callable
import torch

# TYPES
Tensor = torch.Tensor
OptimizerType = Union[torch.optim.Optimizer, List[torch.optim.Optimizer], Tuple[torch.optim.Optimizer, ...], Mapping[
    str, torch.optim.Optimizer]]

# loss functions are torch.nn.Module that
# - in input: (predictions : Tensor[N,*], target : Tensor[N,*] )
# - in output: scalar if reduction is used otherwise Tensor[N,*]
Criterion = Union[torch.nn.Module, Callable[[Tensor, Tensor], Union[float, Tensor]]]

__all__ = ["List", "Tuple", "Mapping", "Union", "Tensor", "BatchTensor", "OptimizerType", "NoReturn"]

# ABSTRACT
from abc import ABC, abstractmethod
