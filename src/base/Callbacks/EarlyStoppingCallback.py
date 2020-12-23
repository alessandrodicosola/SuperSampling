from base.Callbacks import Callback
import math
import base.logging_


class EarlyStoppingCallback(Callback):

    def start_epoch(self, *args, **kwargs):
        if self.capture:
            self.best_loss = kwargs.get('last_loss')
            self.best_epoch = kwargs.get('epoch')
            self._logger.debug("Received from Trainer best_loss: %s and best_epoch: %s", self.best_loss,
                               self.best_epoch)
            self.capture = False

    def end_epoch(self, *args, **kwargs):
        epoch = kwargs.get('epoch')
        val_state = kwargs.get('val_state')
        val_loss = val_state.loss

        if val_loss <= self.best_loss + self.eps:

            self._logger.debug("loss improved from %s to %s : %s -> %s",
                               self.best_epoch, epoch, self.best_loss, val_loss)
            # reset patience counter
            self.counter_patience = self.patience

            self.best_loss = val_loss
            self.best_epoch = epoch

            save_model_fn = kwargs.get('save_model_fn')
            # call Trainer.save_model(loss,epoch)
            save_model_fn(f"{epoch}_{val_loss:.2f}")
        else:
            self.counter_patience -= 1
            self._logger.debug("val_loss not improving.")
            if self.counter_patience == 0:
                stop_fn = kwargs.get('stop_fn')
                print(f"Early stopping at epoch {epoch}. Best epoch: {self.best_epoch}")
                # call Trainer.stop(self)
                stop_fn(self)

    def start_batch(self, *args, **kwargs):
        pass

    def end_batch(self, *args, **kwargs):
        pass

    def __init__(self, patience: int, eps: float = 1e-7):
        super(EarlyStoppingCallback, self).__init__()
        self.patience = patience
        self.counter_patience = patience
        self.eps = eps

        self.best_loss = None
        self.best_epoch = None

        self.capture = True
