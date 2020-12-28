from base.Callbacks.Callback import Callback


class StdOutCallback(Callback):
    """Callback that print train and val metrics at the end of each epoch"""

    def start_epoch(self, *args, **kwargs):
        pass

    def end_epoch(self, *args, **kwargs):
        # for avoiding circular dependency error #
        from base.Trainer import Trainer

        train_state = kwargs.get('train_state')
        val_state = kwargs.get('val_state')

        train_state = Trainer._return_loss_and_metric_formatted(train_state, train=True)
        val_state = Trainer._return_loss_and_metric_formatted(val_state, train=False)

        print("Epoch: ", kwargs.get('epoch') + 1)
        print(train_state)
        print(val_state)

    def start_batch(self, *args, **kwargs):
        pass

    def end_batch(self, *args, **kwargs):
        if self.print_batch:
            from base.Trainer import Trainer

            train_state = kwargs.get('train_state')
            val_state = kwargs.get('val_state')

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
