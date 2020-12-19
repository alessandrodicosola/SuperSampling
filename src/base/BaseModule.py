from __future__ import annotations

import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from pathlib import Path

import utility

class BaseModule(nn.Module, ABC):
    """
    Base class for creating a  neural network
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Base method for inference

        NOTE: Use << with torch.no_grad(): >> when executing validation or testing since there is no need to have
        gradients.

        Args:
            *args: Any kind of input
            **kwargs: Any kind of input

        Returns:
            output
        """
        raise NotImplementedError("forward")

    @abstractmethod
    def train_step(self, *args, **kwargs):
        """Define the training step


        Args:
            *args: can be a list of input
            **kwargs: can be a list of key-value input

        Returns:
            output
        """
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def val_step(self, *args, **kwargs):
        """Define the training step

        NOTE: Use **decorator** <<torch.no_grad()>> when implementing the function since there is no need to have
        gradients.

        Args:
            *args: can be a list of input
            **kwargs: can be a list of key-value input

        Returns:
            output
        """
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def test_step(self, *args, **kwargs):
        """Define a test step

        NOTE: Use **decorator** <<torch.no_grad()>> when implementing the function since there is no need to have
        gradients.

        Args:
            *args: can be a list of input
            **kwargs: can be a list of key-value input

        Returns:
            output
        """
        raise NotImplementedError
