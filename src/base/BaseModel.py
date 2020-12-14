from __future__ import annotations

import torch
import torch.nn as nn

from .hints import Union, NoReturn, Optimizer_

from abc import ABC, abstractmethod
from pathlib import Path

import utility

# TODO: Documentation

class BaseModel(nn.Module, ABC):
    """
    Base class for handling neural networks
    """

    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__()

        # e.g. BaseModel
        self.name: str = self.__class__.__name__
        # e.g. ./models/BaseModels
        self.model_dir: Path = utility.get_models_dir() / self.name

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Base method for inference
        :param args: Any kind of input
        :param kwargs: Any kind of input
        :return: output

        NOTE: Use << with torch.no_grad(): >> when executing validation or testing since there is no need to have
        gradients.
        """
        raise NotImplementedError("forward")

    @abstractmethod
    def train_step(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    @abstractmethod
    def val_step(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    @abstractmethod
    def test_step(self, *args, **kwargs):
        raise NotImplementedError