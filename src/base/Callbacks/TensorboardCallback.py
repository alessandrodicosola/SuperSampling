from datetime import datetime
from typing import List, Tuple

import torch.utils.tensorboard
import torchvision
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import base.logging_
from base.Callbacks import Callback

import math


class TensorboardCallback(Callback):
    """Callback for writing tensorboard summary

    Args:
        log_dir : the root folder where to save events
        print_images : If True images will be written
        print_images_frequency : Set the frequency images are written
        denormalie_fn : function for denormalizing images if they are normalized
    """

    def start_epoch(self, *args, **kwargs):
        self.writer = SummaryWriter(self.log_dir)

    @staticmethod
    def get_plot_train_val(train, val):
        assert len(train) == len(val)

        import matplotlib.pyplot as plt
        import pandas as pd

        smooth = 0.6

        x = range(1, len(train) + 1)
        fig, axis = plt.subplots()
        axis.set_xlabel("Epochs")
        axis.set_ylabel("Loss")

        df_train = pd.DataFrame(train)
        df_val = pd.DataFrame(val)

        axis.plot(x, df_train.ewm(alpha=smooth).mean().values, color="tab:blue", label='train')
        axis.plot(x, df_val.ewm(alpha=smooth).mean().values, color="tab:orange", label='val')

        axis.legend()
        return fig

    def end_epoch(self, *args, **kwargs):
        epoch = kwargs.get('epoch')

        train_state: 'TrainingState' = kwargs.get('train_state')
        for key, value in train_state._asdict().items():
            self.writer.add_scalar(f"Epoch/Train/{key}", value, epoch)

        val_state: 'TrainingState' = kwargs.get('val_state')
        for key, value in val_state._asdict().items():
            self.writer.add_scalar(f"Epoch/Val/{key}", value, epoch)

        self.train_losses.append(train_state.loss)
        self.val_losses.append(val_state.loss)

        self.writer.add_figure("Epoch/TrainVal",
                               TensorboardCallback.get_plot_train_val(self.train_losses, self.val_losses), epoch)

        self.writer.add_scalar("Hyperparamters/lr", kwargs.get('lr'), epoch)
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
            nrow = math.ceil(math.sqrt(kwargs.get('batch_size')))

            # TODO Check this
            if 'val_state' in kwargs and kwargs.get('current_epoch') == 0 or ((kwargs.get('current_epoch') + 1) % self.print_images_frequency) == 0:
                X = kwargs.get("in_")
                if isinstance(X, (List, Tuple)):
                    for index, input in enumerate(filter(lambda elem: isinstance(elem, Tensor), X)):
                        # grid_in = torchvision.utils.make_grid(input.detach(), nrow=4)
                        # self.writer.add_image(f'Epoch/Images/Input{index}_original', grid_in,
                        #                       kwargs.get('current_epoch'))
                        grid_in = torchvision.utils.make_grid(self.get_tensor(input), nrow=nrow).clamp(0, 1)
                        self.writer.add_image(f'Epoch/Images/Input{index}', grid_in, kwargs.get('current_epoch'))
                    del X
                elif isinstance(X, Tensor):
                    # grid_in = torchvision.utils.make_grid(X.detach(), nrow=4)
                    # self.writer.add_image(f'Epoch/Images/Input_original', grid_in, kwargs.get('current_epoch'))
                    grid_in = torchvision.utils.make_grid(self.get_tensor(X), nrow=nrow).clamp(0, 1)
                    self.writer.add_image('Epoch/Images/Input', grid_in, kwargs.get('current_epoch'))
                    del X
                else:
                    raise RuntimeError("Unknown type.")

                grid_out = torchvision.utils.make_grid(self.get_tensor(kwargs.get('out')), nrow=nrow).clamp(0, 1)

                self.writer.add_image('Epoch/Images/Output', grid_out, kwargs.get('current_epoch'))

    def __init__(self, log_dir: str, print_images=False, print_images_frequency: int = 10, denormalize_fn=None):
        super(TensorboardCallback, self).__init__()
        self.log_dir = log_dir
        self.writer: SummaryWriter = None

        self.denormalize_fn = denormalize_fn
        self.print_images = print_images
        self.print_images_frequency = print_images_frequency

        # TODO: Find a better way
        self.train_losses = []
        self.val_losses = []

    def get_tensor(self, tensor: Tensor):
        return self.denormalize_fn(tensor) if self.denormalize_fn else tensor
