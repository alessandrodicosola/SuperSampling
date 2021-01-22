import torch

from base.Callbacks.Callback import Callback, CallbackWrapper


class StdOutCallback(Callback):
    """Callback that print train and val metrics at the end of each epoch"""

    def start_epoch(self, wrapper: CallbackWrapper):
        self.start.record()

    def end_epoch(self, wrapper: CallbackWrapper):
        # for avoiding circular dependency error #
        from base.Trainer import Trainer

        self.end.record()

        train_state = wrapper.train_state
        val_state = wrapper.val_state

        train_state = Trainer._return_loss_and_metric_formatted(train_state, train=True)
        val_state = Trainer._return_loss_and_metric_formatted(val_state, train=False)

        torch.cuda.synchronize()

        elapsed_time_seconds = self.start.elapsed_time(self.end) // 1000
        iteration = f"{wrapper.epoch + 1}/{wrapper.epochs}"
        msg = f"{iteration} (elapsed time (s): {elapsed_time_seconds})"
        print(msg)
        print(train_state)
        print(val_state)

    def start_batch(self, wrapper: CallbackWrapper):
        pass

    def end_batch(self, wrapper: CallbackWrapper):
        if self.print_batch:
            from base.Trainer import Trainer

            train_state = wrapper.train_state
            val_state = wrapper.val_state

            if train_state:
                assert val_state is None
                train_state = Trainer._return_loss_and_metric_formatted(train_state, train=True)
                print(train_state)

            if val_state:
                assert train_state is None
                val_state = Trainer._return_loss_and_metric_formatted(val_state, train=False)
                print(val_state)

    def __init__(self, print_batch=False):
        super(StdOutCallback, self).__init__()
        self.print_batch = print_batch

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpickable element
        del state['start']
        del state['end']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add unpickable element
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

