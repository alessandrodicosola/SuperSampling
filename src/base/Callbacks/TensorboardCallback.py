from datetime import datetime

import torch.utils.tensorboard
import torchvision
from torch import Tensor

import base.logging_
from base.Callbacks import Callback


class TensorboardCallback(Callback):
    """Callback for writing tensorboard summary

    Args:
        log_dir : the root folder where to save events
        print_images : If True images will be written
        print_images_frequency : Set the frequency images are written
        denormalie_fn : function for denormalizing images if they are normalized
    """

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

        global_step = kwargs.get('batch_index') + (kwargs.get('current_epoch') * kwargs.get('batch_nums'))

        train_state = kwargs.get('train_state', None)
        val_state = kwargs.get('val_state', None)
        if train_state is None and val_state is None:
            raise RuntimeError("Both train_state and val_state are None type!")

        if train_state:
            assert val_state is None, 'val_state is not None.'
            for key, value in train_state._asdict().items():
                self.writer.add_scalar(f"Batches/Train/{key}", value, global_step)

        if val_state:
            assert train_state is None, 'train_state is not None.'
            for key, value in val_state._asdict().items():
                self.writer.add_scalar(f"Batches/Val/{key}", value, global_step)

        if self.print_images:
            if kwargs.get('current_epoch') % self.print_images_frequency == 0 and 'val_state' in kwargs:
                grid_in = torchvision.utils.make_grid(self.get_tensor(kwargs.get('in_')), nrow=3)
                grid_out = torchvision.utils.make_grid(self.get_tensor(kwargs.get('out')), nrow=3)

                self.writer.add_image('Epoch/Images/Input', grid_in, kwargs.get('current_epoch'))
                self.writer.add_image('Epoch/Images/Output', grid_out, kwargs.get('current_epoch'))

    def __init__(self, log_dir: str, print_images=False, print_images_frequency: int = 10, denormalize_fn=None):
        super(TensorboardCallback, self).__init__()
        self.log_dir = log_dir + "/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")
        self.writer = None

        self.denormalize_fn = None
        self.print_images = print_images
        self.print_images_frequency = print_images_frequency

    def get_tensor(self, tensor: Tensor):
        return self.denormalize_fn(tensor) if self.denormalize_fn else tensor
