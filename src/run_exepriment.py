# python file where train and test command will be run
import logging
import random
from functools import partial

import torch
import numpy as np
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader

from base.Callbacks.TensorboardCallback import TensorboardCallback
from base.Trainer import Trainer
from base.metrics.PSNR import PSNR
from base.metrics.SSIM import SSIM
from datasets.ASDNDataset import ASDNDataset, collate_fn, NormalizeInverse
import models.ASDN

from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation
from utility import get_datasets_dir


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


def fix_randomness(seed: int):
    """Fix all the possible sources of randomness.

        Args:
            seed: the seed to use.
        """
    # Add comment for each random seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run(experiment: str, n_workers: int, pin_memory: bool, epochs: int, batch_size: int, save_checkpoints: int = 1, **model_kwargs):
    """Run the experiment with specified epochs

    Args:
        experiment:
        n_workers:
        pin_memory:
        epochs:
        batch_size:

    Keyword Args:
        in_channels
        low_level_features
        n_dab
        out_channels_dab
        intra_layer_output_features
        n_intra_layers
        reduction

    Returns:
        error : bool
    """
    # set experiment string
    models.ASDN.set_save_checkpoints(save_checkpoints)
    experiment += f"_E{epochs}_B{batch_size}_S{models.ASDN._SEGMENTS_GRADIENT_CHECKPOINT}"
    if len(model_kwargs) > 0:
        experiment += "_".join(map(lambda elem: f"{elem[0]}{elem[1]}", model_kwargs.items()))

    print(f"Experimenting with: {experiment}")

    # Get the device
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. Cannot train the network.")

    DEVICE = torch.device('cuda:0')

    # Greater than 8 led to OutOfMemory using default configuration
    BATCH_SIZE = batch_size
    # BUG: https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
    # PyTorch spawn process until crash n both cpu or gpu (if pin_memory = True )
    # python multiprocessing has problems in windows
    # has workaround I'm using FastDataLoader
    NUM_WORKERS = n_workers
    PIN_MEMORY = pin_memory

    PATCH_SIZE = 48

    # Get the laplacian frequency representation between 1 and 2
    LFR = LaplacianFrequencyRepresentation(1, 2, 11)

    # Define the collate function
    ASDN_COLLATE_FN = partial(collate_fn, lfr=LFR)

    # Prepare datasets
    DATASET_DIR = get_datasets_dir() / "DIV2K"
    TRAIN_DATASET = ASDNDataset(DATASET_DIR / "DIV2K_train_HR", patch_size=PATCH_SIZE, lfr=LFR, augmentation=None)
    TRAIN_DATALOADER = FastDataLoader(dataset=TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS,
                                      collate_fn=ASDN_COLLATE_FN, pin_memory=PIN_MEMORY)

    VAL_DATASET = ASDNDataset(DATASET_DIR / "DIV2K_valid_HR", patch_size=PATCH_SIZE, lfr=LFR, augmentation=None)
    VAL_DATALOADER = FastDataLoader(dataset=VAL_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                    collate_fn=ASDN_COLLATE_FN, pin_memory=PIN_MEMORY)

    # Prepare model with default parameters
    ASDN_ = models.ASDN.ASDN(input_image_channels=3, lfr=LFR, **model_kwargs).to(DEVICE)

    # lr=3e-4 as suggested in https://karpathy.github.io/2019/04/25/recipe/
    ADAM = Adam(ASDN_.parameters(), lr=3e-4, betas=(0.9, 0.999))

    # Define the criterion
    L1 = L1Loss(reduction='mean').to(DEVICE)

    # Define the metrics
    # SSIM cause out of memory both on GPU and on CPU
    METRICS = PSNR(max_pixel_value=1.0, reduction='mean')  # SSIM(max_pixel_value=1.0, reduction='mean')]

    # Define the trainer
    TRAINER = Trainer(experiment=experiment, model=ASDN_, optimizer=ADAM, criterion=L1, metric=METRICS,
                      device=DEVICE,
                      lr_scheduler=None,
                      callback=None)

    tensorboard_callback = TensorboardCallback(log_dir=str(TRAINER.log_dir), print_images=True,
                                               print_images_frequency=10,
                                               denormalize_fn=NormalizeInverse(mean=TRAIN_DATASET.mean,
                                                                               std=TRAIN_DATASET.std))

    TRAINER.callback.add_callback(tensorboard_callback)

    error: bool = False
    try:
        TRAINER.fit(train_loader=TRAIN_DATALOADER, val_loader=VAL_DATALOADER, epochs=epochs)
    except (RuntimeError, ValueError) as e:
        tensorboard_callback.writer.add_text("Error", str(e))
        error = True
    finally:
        # cleanup
        del TRAINER
        del ADAM
        del ASDN_

        tensorboard_callback.writer.close()
        del tensorboard_callback

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return error


if __name__ == "__main__":
    run("BATCH_TIME", n_workers=1,pin_memory=True,epochs=1,batch_size=8, n_dab=5,n_intra_layers=4,out_channels_dab=56, save_checkpoints=1)

