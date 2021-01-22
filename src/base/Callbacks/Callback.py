import logging

logging.basicConfig()

from abc import ABC, abstractmethod
from typing import List

class CallbackWrapper:
    """ Class for containing key=value pair using OOP

    Args:
        kwargs: dictionary of key-value pairs

    Keyword Args:
        epochs : int
        epoch  : int

        trainer : Trainer
        train_state: TrainingState
        val_state : TrainingState

        batch_index : int
        batch_nums : int
        batch_size : int

        last_val_loss : float

    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, item):
        return self.__dict__.get(item) if item in self.__dict__ else None


class Callback(ABC):
    """Define common operations to do before,after each epoch or batch"""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG)

    @abstractmethod
    def start_epoch(self, wrapper: CallbackWrapper):
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
    def end_epoch(self, wrapper: CallbackWrapper):
        """Operation to do after train and validation in an epoch

        Keyword Args:
            resuming
            epoch
            epochs
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
    def start_batch(self, wrapper: CallbackWrapper):
        """Operation to do before processing the batch

        Keyword Args:
            batch_index
            batch_size
            batch_num
            current_epoch

            # common args
            stop_fn: Pointer to the function for stopping the training
            log_dir: directory for logging
            optimizer: torch.optim.Optimizer used
            train: True: model in training mode

        """
        raise NotImplementedError

    @abstractmethod
    def end_batch(self, wrapper: CallbackWrapper):
        """Operation to do after the forwarding (and evaluation of the loss and metrics) of the batch

        Keyword Args:
            batch_index
            batch_size
            batch_num
            current_epoch
            train_state | val_state ('TrainingState' namedtuple with loss and metrics)

            # common args
            stop_fn: Pointer to the function for stopping the training
            log_dir: directory for logging
            optimizer: torch.optim.Optimizer used
            train: True: model in training mode

            # for testing during training
            in_ : input batch as Tensor
            out : output batch as Tensor

        """
        raise NotImplementedError


class ListCallback(Callback):

    def __init__(self):
        super(ListCallback, self).__init__()
        self.callbacks = list()

    def start_epoch(self, wrapper: CallbackWrapper):
        for callback in self.callbacks:
            callback.start_epoch(wrapper)

    def end_epoch(self, wrapper: CallbackWrapper):
        for callback in self.callbacks:
            callback.end_epoch(wrapper)

    def start_batch(self, wrapper: CallbackWrapper):
        for callback in self.callbacks:
            callback.start_batch(wrapper)

    def end_batch(self, wrapper: CallbackWrapper):
        for callback in self.callbacks:
            callback.end_batch(wrapper)

    def add_callback(self, callback: Callback):
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callback):
        self.callbacks.remove(callback)

    @classmethod
    def from_list(cls, callbacks: List[Callback]):
        obj = cls()
        obj.callbacks.extend(callbacks)
        return obj
