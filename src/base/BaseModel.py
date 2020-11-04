import torch
import torch.nn as nn
import utility
from abc import ABC,abstractmethod
from pathlib import Path


class BaseModel(nn.Module, ABC):
    """
    Base class for handling neural networks
    """
    def __init__(self):
        super(BaseModel,self).__init__()
        #e.g. BaseModel
        self.name : str = self.__class__.__name__
        #e.g. ./models/BaseModels
        self.model_dir : Path = utility.get_models_dir() / self.name

    @abstractmethod
    def forward(self,input):
        """
        Base method for inference phase
        :param input: input
        :return: output

        NOTE: Use << with torch.no_grad(): >> when executing validation or testing since there is no need to have gradients.
        """
        raise NotImplementedError("forward")


    @abstractmethod
    def train_step(self,batch,batch_index):
        """
        Implement training step called by [BaseTrainer]
        :param batch: batch
        :param batch_index: batch index
        :return: training information

        ATTENTION: must call super().train_step(batch,batch_index) otherwise unexpected result will be seen during training
        """

        # set the model in training phase
        self.train()

        pass

    @abstractmethod
    @torch.no_grad()
    def val_step(self,batch,batch_index):
        """
        Implement validation step called by [BaseTrainer]
        :param batch: batch
        :param batch_index: batch index
        :return: training information

        NOTE: @torch.no_grad() avoid to compute gradients.
        ATTENTION: must call super().val_step(batch,batch_index) otherwise unexpected result will be seen during validation
        """

        #set the model in validation phase
        self.eval()

        pass