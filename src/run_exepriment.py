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
from datasets.ASDNDataset import ASDNDataset, collate_fn
from models.ASDN import ASDN
from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation
from utility import get_datasets_dir

import base.logging_


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


def main(experiment: str, epochs: int, batch_size: int):
    """Run the exepriment with specified epochs

    Args:
        experiment: name of the experiment
        epochs: epochs to run

    Returns:
        history of the training ( Since tensorboard callback is used is useless)

    """
    # Get the device
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. Can train the network.")
    DEVICE = torch.device('cuda:0')

    # Greater than 8 led to OutOfMemory
    BATCH_SIZE = batch_size
    NUM_WORKERS = 4
    PIN_MEMORY = True

    PATCH_SIZE = 48

    # Get the laplacian frequency representation between 1 and 2
    LFR = LaplacianFrequencyRepresentation(1, 2, 11)

    # Define the collate function
    ASDN_COLLATE_FN = partial(collate_fn, lfr=LFR)

    # Prepare datasets
    DATASET_DIR = get_datasets_dir() / "DIV2K"
    TRAIN_DATASET = ASDNDataset(DATASET_DIR / "DIV2K_train_HR", patch_size=PATCH_SIZE, lfr=LFR, augmentation=None)
    TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                  collate_fn=ASDN_COLLATE_FN, pin_memory=PIN_MEMORY)

    VAL_DATASET = ASDNDataset(DATASET_DIR / "DIV2K_train_HR", patch_size=PATCH_SIZE, lfr=LFR, augmentation=None)
    VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                collate_fn=ASDN_COLLATE_FN, pin_memory=PIN_MEMORY)

    # Prepare model with default parameters
    ASDN_ = ASDN(input_image_channels=3, lfr=LFR).to(DEVICE)

    # Define the optimizer Adam with default parameters (although lr should be 1e-4 in order to be default by mean of the paper)
    ADAM = Adam(ASDN_.parameters(), lr=1e-3, betas=(0.9, 0.999))

    # NOTE: Use reduction='sum' since the trainer divided by total_samples

    # Define the criterion
    L1 = L1Loss(reduction='sum').to(DEVICE)

    # Define the metrics
    METRICS = [PSNR(reduction='sum').to(DEVICE), SSIM(reduction='sum').to(DEVICE)]

    # Define the trainer
    TRAINER = Trainer(experiment=experiment, model=ASDN_, optimizer=ADAM, criterion=L1, metric=METRICS, device=DEVICE,
                      lr_scheduler=None, callback=None)

    # Add Tensorboard callback external the init because it needs log_dir
    TRAINER.callback.add_callback(
        TensorboardCallback(log_dir=str(TRAINER.log_dir), print_images=True, print_images_frequency=10))

    return TRAINER.fit(train_loader=TRAIN_DATALOADER, val_loader=VAL_DATALOADER, epochs=epochs)


if __name__ == "__main__":
    base.logging_.disable_up_to(logging.ERROR)

    import argparse

    parser = argparse.ArgumentParser("run_experiment.py")
    parser.add_argument("experiment", type=str)
    parser.add_argument("epochs", type=int)
    parser.add_argument("batch_size", type=int)

    result = parser.parse_args()

    experiment = result.experiment
    epochs = result.epochs
    batch_size = result.batch_size

    main(experiment=experiment, epochs=epochs, batch_size=batch_size)
