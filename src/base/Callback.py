from .hints import List
from abc import ABC, abstractmethod


class Callback(ABC):
    """
    Base class for handling operations in the following situation:
     - START EPOCH
     - START BATCH
     - END BATCH
     - END EPOCH
    for both training and tuning (validation)
    """

    @abstractmethod
    def start_epoch(self, *args, **kwargs):
        raise NotImplementedError("start_epoch")

    @abstractmethod
    def end_epoch(self, *args, **kwargs):
        raise NotImplementedError("end_epoch")


class ListCallback(Callback):

    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def start_epoch(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.start_epoch(*args, **kwargs)

    def end_epoch(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.end_epoch(*args, **kwargs)
