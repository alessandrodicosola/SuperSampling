from base.Callbacks import Callback
import math
import base.logging_
from base.Callbacks.Callback import CallbackWrapper


class EarlyStoppingCallback(Callback):

    def start_epoch(self, wrapper: CallbackWrapper):
        if self.capture:
            self.best_loss = wrapper.last_val_loss
            self.best_epoch = wrapper.epoch
            self._logger.debug("Received from Trainer best_loss: %s and best_epoch: %s", self.best_loss,
                               self.best_epoch)
            self.capture = False

    def end_epoch(self, wrapper: CallbackWrapper):
        epoch = wrapper.epoch
        val_loss = wrapper.val_state.loss

        if val_loss <= self.best_loss + self.eps:

            self._logger.debug("loss improved from %s to %s : %s -> %s",
                               self.best_epoch, epoch, self.best_loss, val_loss)
            # reset patience counter
            self.counter_patience = self.patience

            self.best_loss = val_loss
            self.best_epoch = epoch

            # call Trainer.save_model(loss,epoch)
            wrapper.trainer.save_model(f"{epoch}_{val_loss:.2f}")
        else:
            self.counter_patience -= 1
            self._logger.debug("val_loss not improving.")
            if self.counter_patience == 0:
                print(f"Early stopping at epoch {epoch}. Best epoch: {self.best_epoch}")
                wrapper.trainer._stop_fn(self)

    def start_batch(self, wrapper: CallbackWrapper):
        pass

    def end_batch(self, wrapper: CallbackWrapper):
        pass

    def __init__(self, patience: int, eps: float = 1e-7):
        super(EarlyStoppingCallback, self).__init__()
        self.patience = patience
        self.counter_patience = patience
        self.eps = eps

        self.best_loss = None
        self.best_epoch = None

        self.capture = True
