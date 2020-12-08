# A file for define types for all the project

from pathlib import Path

from typing import List, Tuple, Mapping, Union, NoReturn
import torch

# TYPES
BatchTensor = Union[torch.Tensor,List[torch.Tensor],Tuple[torch.Tensor]]
OptimizerType  = Union[torch.optim.Optimizer,List[torch.optim.Optimizer],Tuple[torch.optim.Optimizer,...],Mapping[str,torch.optim.Optimizer]]

__all__ = ["List","Tuple","Mapping","Union","BatchTensor","OptimizerType","NoReturn"]


# ABSTRACT
from abc import ABC,abstractmethod
