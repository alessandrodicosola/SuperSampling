import torch.nn as nn
import utility
from abc import ABC,abstractmethod
from pathlib import Path
class BaseModel(nn.Module, ABC):
    """
    Base class for handling NNs
    """
    def __init__(self):
        super(BaseModel,self).__init__()

        self.name : str = self.__class__.__name__
        self.model_dir : Path = utility.get_models_dir() / self.name


    @abstractmethod
    def forward(self,input):
        raise NotImplementedError("forward")
