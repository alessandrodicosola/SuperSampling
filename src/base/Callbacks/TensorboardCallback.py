from datetime import datetime

import torch.utils.tensorboard

from base.Callbacks import Callback


class TensorboardCallback(Callback):
    """Callback for writing tensorboard summary"""

    def start_epoch(self, *args, **kwargs):
        pass

    def end_epoch(self, *args, **kwargs):
        epoch = kwargs.get('epoch')

        train_state: 'TrainingState' = kwargs.get('train_state')
        for key, value in train_state._asdict().items():
            self.writer.add_scalar(f"Epoch/Train/{key}", value, epoch)

        val_state: 'TrainingState' = kwargs.get('val_state')
        for key, value in val_state._asdict().items():
            self.writer.add_scalar(f"Epoch/Train/{key}", value, epoch)

        self.writer.add_scalar(f"Epoch/lr", kwargs.get('optimizer').param_groups[0]['lr'], epoch)
        self.writer.flush()

    def start_batch(self, *args, **kwargs):
        pass

    def end_batch(self, *args, **kwargs):
        pass

    def __init__(self, log_dir: str):
        super(TensorboardCallback, self).__init__()
        log_dir += "/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")
        self.writer = torch.utils.tensorboard.SummaryWriter(log_dir)
