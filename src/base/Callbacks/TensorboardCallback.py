from datetime import datetime

import torch.utils.tensorboard

import base.logging_
from base.Callbacks import Callback


class TensorboardCallback(Callback):
    """Callback for writing tensorboard summary"""

    def start_epoch(self, *args, **kwargs):
        self.writer = torch.utils.tensorboard.SummaryWriter(self.log_dir)

    def end_epoch(self, *args, **kwargs):
        epoch = kwargs.get('epoch')

        train_state: 'TrainingState' = kwargs.get('train_state')
        for key, value in train_state._asdict().items():
            self.writer.add_scalar(f"Epoch/Train/{key}", value, epoch)

        val_state: 'TrainingState' = kwargs.get('val_state')
        for key, value in val_state._asdict().items():
            self.writer.add_scalar(f"Epoch/Val/{key}", value, epoch)

        self.writer.add_scalar(f"Hyperparamters/lr", kwargs.get('optimizer').param_groups[0]['lr'], epoch)
        self.writer.close()

    def start_batch(self, *args, **kwargs):
        pass

    def end_batch(self, *args, **kwargs):
        self._logger.debug("%s  %s    %s", kwargs.get('batch_index'), kwargs.get('current_epoch'),
                           kwargs.get('batch_num'))

        global_step = kwargs.get('batch_index') + (kwargs.get('current_epoch') * kwargs.get('batch_num'))

        train_state = kwargs.get('train_state', None)
        val_state = kwargs.get('val_state', None)
        if train_state is None and val_state is None:
            raise RuntimeError("Both train_state and val_state are None type!")

        if train_state:
            for key, value in train_state._asdict().items():
                self.writer.add_scalar(f"Batches/Train/{key}", value, global_step)

        if val_state:
            for key, value in val_state._asdict().items():
                self.writer.add_scalar(f"Batches/Val/{key}", value, global_step)

    def __init__(self, log_dir: str):
        super(TensorboardCallback, self).__init__()
        self.log_dir = log_dir + "/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")
        self.writer = None
