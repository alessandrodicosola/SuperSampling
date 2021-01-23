from datetime import datetime
from typing import List, Tuple

import torch.utils.tensorboard
import torchvision
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import base.logging_
from base.Callbacks import Callback

import math

from base.Callbacks.Callback import CallbackWrapper


class TensorboardCallback(Callback):
    """Callback for writing tensorboard summary

    Args:
        log_dir : the root folder where to save events
        print_images : If True images will be written
        print_images_frequency : Set the frequency images are written
        denormalie_fn : function for denormalizing images if they are normalized
    """

    def start_epoch(self, wrapper: CallbackWrapper):
        self.writer = SummaryWriter(self.log_dir)

    @staticmethod
    def get_plot_train_val(train, val, alpha=0.3):
        assert len(train) == len(val)

        import matplotlib.pyplot as plt

        import pandas as pd

        df = pd.DataFrame(zip(train, val), columns=["train", "val"])

        fig, axis = plt.subplots()

        color = "tab:orange"
        axis.plot(df.train.values, alpha=0.25, color=color)
        axis.plot(df.train.ewm(alpha=alpha).mean().values, color=color, linewidth=2, label="train")

        color = "tab:blue"
        axis.plot(df.val.values, alpha=0.25, color=color)
        axis.plot(df.val.ewm(alpha=alpha).mean().values, color=color, linewidth=2, label="val")
        axis.grid(alpha=0.3)

        axis.legend()
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        fig.tight_layout()
        return fig

    @staticmethod
    def history_to_train_val_loss(history: 'HistoryState'):
        train = list(map(lambda state: state.loss, history.train))
        val = list(map(lambda state: state.loss, history.val))
        return train, val

    def end_epoch(self, wrapper: CallbackWrapper):
        epoch = wrapper.epoch

        train_state = wrapper.train_state
        for key, value in train_state._asdict().items():
            self.writer.add_scalar(f"Epoch/Train/{key}", value, epoch)

        val_state = wrapper.val_state
        for key, value in val_state._asdict().items():
            self.writer.add_scalar(f"Epoch/Val/{key}", value, epoch)

        train, val = TensorboardCallback.history_to_train_val_loss(wrapper.history)

        self.writer.add_figure("Epoch/TrainVal",
                               TensorboardCallback.get_plot_train_val(train, val), epoch)

        self.writer.add_scalar("Hyperparamters/lr", wrapper.trainer.optimizer.param_groups[0]['lr'], epoch)
        self.writer.close()

    def start_batch(self, wrapper: CallbackWrapper):
        pass

    def end_batch(self, wrapper: CallbackWrapper):
        current_epoch = wrapper.epoch
        batch_index = wrapper.batch_index
        batch_nums = wrapper.batch_nums
        batch_size = wrapper.batch_size

        global_step = batch_index + current_epoch * batch_nums

        train_state = wrapper.train_state
        val_state = wrapper.val_state

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
            nrow = math.ceil(math.sqrt(batch_size))

            if val_state and (
                    current_epoch == 0 or (((current_epoch + 1) % self.print_images_frequency) == 0)):

                X = wrapper.X

                if isinstance(X, (List, Tuple)):
                    for index, input in enumerate(filter(lambda elem: isinstance(elem, Tensor), X)):
                        # grid_in = torchvision.utils.make_grid(input.detach(), nrow=4)
                        # self.writer.add_image(f'Epoch/Images/Input{index}_original', grid_in,
                        #                       kwargs.get('current_epoch'))
                        grid_in = torchvision.utils.make_grid(self.get_tensor(input), nrow=nrow)
                        self.writer.add_image(f'Epoch/Images/Input{index}', grid_in, current_epoch)
                elif isinstance(X, Tensor):
                    # grid_in = torchvision.utils.make_grid(X.detach(), nrow=4)
                    # self.writer.add_image(f'Epoch/Images/Input_original', grid_in, kwargs.get('current_epoch'))
                    grid_in = torchvision.utils.make_grid(self.get_tensor(X), nrow=nrow)
                    self.writer.add_image('Epoch/Images/Input', grid_in, current_epoch)
                else:
                    raise RuntimeError(
                        f"Unknown type. received: {type(X)}, expected: Tensor, List[Tensor], Tuple[Tensor].")

                grid_out = torchvision.utils.make_grid(self.get_tensor(wrapper.out), nrow=nrow)
                grid_ground_truth = torchvision.utils.make_grid(self.get_tensor(wrapper.ground_truth), nrow=nrow)
                self.writer.add_image('Epoch/Images/Output', grid_out, current_epoch)
                self.writer.add_image('Epoch/Images/Ground Truth', grid_ground_truth, current_epoch)

    def __init__(self, log_dir: str, print_images=False, print_images_frequency: int = 10, denormalize_fn=None):
        super(TensorboardCallback, self).__init__()
        self.log_dir = log_dir
        self.writer: SummaryWriter = None

        self.denormalize_fn = denormalize_fn
        self.print_images = print_images
        self.print_images_frequency = print_images_frequency

    def get_tensor(self, tensor: Tensor):
        return self.denormalize_fn(tensor) if self.denormalize_fn else tensor
