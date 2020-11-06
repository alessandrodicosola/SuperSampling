import torch
import torch.nn as nn
import utility

from typing import List,Tuple,Mapping,Union

from abc import ABC, abstractmethod
from pathlib import Path

class BaseModel(nn.Module, ABC):
    """
    Base class for handling neural networks
    """

    def __init__(self):
        super(BaseModel, self).__init__()

        # e.g. BaseModel
        self.name: str = self.__class__.__name__
        # e.g. ./models/BaseModels
        self.model_dir: Path = utility.get_models_dir() / self.name

        # retrieve the optimizer
        self.optimizer : torch.optim.Optimizer = self.get_optimizer()


    @abstractmethod
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        """
        Base method for inference phase
        :param input: input
        :return: output

        NOTE: Use << with torch.no_grad(): >> when executing validation or testing since there is no need to have gradients.
        """
        raise NotImplementedError("forward")

    @abstractmethod
    def train_step(self, batch : Union[Tuple[torch.Tensor,...],torch.Tensor], batch_index: int):
        """
        Implement training step called by [BaseTrainer]
        :param batch: batch
        :param batch_index: batch index
        :return: loss (for doing loss.backwards()) and other information

        ATTENTION: must call super().train_step(batch,batch_index) otherwise unexpected result will be seen during training
        """

        # set the model in training phase
        self.train()

        pass

    @abstractmethod
    @torch.no_grad()
    def val_step(self, batch : Union[Tuple[torch.Tensor,...],torch.Tensor], batch_index : int):
        """
        Implement validation step called by [BaseTrainer]
        :param batch: batch
        :param batch_index: batch index
        :return: :return: loss and other information

        NOTE: @torch.no_grad() avoid to compute gradients.
        ATTENTION: must call super().val_step(batch,batch_index) otherwise unexpected result will be seen during validation
        """

        # set the model in validation phase
        self.eval()

        pass

    @abstractmethod
    def get_optimizers(self) -> Union[torch.optim.Optimizer,List[torch.optim.Optimizer],Mapping[str,torch.optim.Optimizer],Tuple[torch.optim.Optimizer,...]]:
        """
        Function for retrieving the optimizer(s) necessary during the train_step and val_step
        :param optimizer: to
        :param kwargs:
        :return: optimizer or list of optimizer or dict of optimizer
        """
        raise NotImplementedError("get_optimizer")



