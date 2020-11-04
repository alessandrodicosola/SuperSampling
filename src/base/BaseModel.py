import torch
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
        #e.g. BaseModel
        self.name : str = self.__class__.__name__
        #e.g. ./models/BaseModels
        self.model_dir : Path = utility.get_models_dir() / self.name

    @torch.no_grad()
    @abstractmethod
    def forward(self,input):
        """
        Base method for inference phase
        :param input: input
        :return: output

        NOTE FROM THE DOCUMENTATION:
        @torch.no_grad()
        Disabling gradient calculation is useful for inference,
        when you are sure that you will not call Tensor.backward().
        It will reduce memory consumption for computations that would otherwise have requires_grad=True.
        """
        raise NotImplementedError("forward")

    def train_step(self,batch,batch_index):
        """
        Implement training step called by [BaseTrainer]
        :param batch: batch
        :param batch_index: batch index
        :return: training information
        """

        # set the model in training phase
        self.train()
        raise NotImplementedError("train_step")

    @torch.no_grad()
    def val_step(self,batch,batch_index):
        """
        Implement training step called by [BaseTrainer]
        :param batch: batch
        :param batch_index: batch index
        :return: training information

        NOTE: @torch.no_grad() avoid to compute gradients.
        """

        #set the model in validation phase
        self.eval()
        raise NotImplementedError("val_step")
