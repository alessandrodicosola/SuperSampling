from base.Callbacks import Callback


class CheckpointCallback(Callback):
    def start_epoch(self, *args, **kwargs):
        pass

    def end_epoch(self, *args, **kwargs):
        epoch = kwargs.get("epoch")
        if epoch % self.frequency:
            save_model_fn = kwargs.get("save_model_fn")
            save_model_fn(f"{epoch}.checkpoint")

    def start_batch(self, *args, **kwargs):
        pass

    def end_batch(self, *args, **kwargs):
        pass

    def __init__(self, frequency):
        super(CheckpointCallback, self).__init__()
        self.frequency = frequency
