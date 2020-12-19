import logging

logging.basicConfig()

from abc import ABC, abstractmethod
from typing import List


class Callback(ABC):
    """Define common operations to do before,after each epoch or batch"""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG)

    @abstractmethod
    def start_epoch(self, *args, **kwargs):
        """Operation to do before train and validation in an epoch

        Keyword Args:
            resuming
            epoch
            last_loss

            # common args
            stop_fn: Pointer to the function for stopping the training
            log_dir: directory for logging
            optimizer: torch.optim.Optimizer used
            train: True: model in training mode
        """
        raise NotImplementedError

    @abstractmethod
    def end_epoch(self, *args, **kwargs):
        """Operation to do after train and validation in an epoch

        Keyword Args:
            resuming
            epoch
            train_state
            val_state
            current_history

            # common args
            stop_fn: Pointer to the function for stopping the training
            log_dir: directory for logging
            optimizer: torch.optim.Optimizer used
            train: True: model in training mode
        """
        raise NotImplementedError

    @abstractmethod
    def start_batch(self, *args, **kwargs):
        """Operation to do before processing the batch

        Keyword Args:
            batch_index
            batch_size
            epoch

            # common args
            stop_fn: Pointer to the function for stopping the training
            log_dir: directory for logging
            optimizer: torch.optim.Optimizer used
            train: True: model in training mode

        """
        raise NotImplementedError

    @abstractmethod
    def end_batch(self, *args, **kwargs):
        """Operation to do after the forwarding (and evaluation of the loss and metrics) of the batch

        Keyword Args:
            batch_index
            batch_size
            epoch
            batch_metrics (namedtuple with loss and metrics)

            # common args
            stop_fn: Pointer to the function for stopping the training
            log_dir: directory for logging
            optimizer: torch.optim.Optimizer used
            train: True: model in training mode

        """
        raise NotImplementedError


class ListCallback(Callback):

    def __init__(self):
        super(ListCallback, self).__init__()
        self.callbacks = list()

    def start_epoch(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.start_epoch(*args, **kwargs)

    def end_epoch(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.end_epoch(*args, **kwargs)

    def start_batch(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.start_batch(*args, **kwargs)

    def end_batch(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.end_batch(*args, **kwargs)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def remove_callback(self, callback):
        self.callbacks.remove(callback)

    @classmethod
    def from_list(cls, callbacks: List[Callback]):
        obj = cls()
        obj.callbacks.extend(callbacks)
        return obj
