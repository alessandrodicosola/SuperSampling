from collections import namedtuple

from base.Callbacks import Callback
from base.Callbacks.Callback import CallbackWrapper


class CheckpointCallback(Callback):
    def start_epoch(self, wrapper: CallbackWrapper):
        pass

    def end_epoch(self, wrapper: CallbackWrapper):
        epoch = wrapper.current_epoch

        if epoch == 0 or epoch + 1 % self.frequency == 0:
            wrapper.trainer.save_model_fn(f"{epoch}.checkpoint")

    def start_batch(self, wrapper: CallbackWrapper):
        pass

    def end_batch(self, wrapper: CallbackWrapper):
        pass

    def __init__(self, frequency):
        super(CheckpointCallback, self).__init__()
        self.frequency = frequency
