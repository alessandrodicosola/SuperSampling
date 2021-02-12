import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Tuple, List

import torch
import torchvision
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from base import BaseModule
from base.trainer.functional import move_to_device, forward, get_batch_size
from base.transforms.RandomCompose import RandomCompose
from base.transforms.Rotate import Rotate
from datasets.ASDNDataset import create_batch_for_training, ASDNDataset
from models.ASDN import ASDN
from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation
from run_exepriment import FastDataLoader, fix_randomness
from utility import get_datasets_dir, get_models_dir


class SnapshotEnsembleLRScheduler(torch.optim.lr_scheduler.LambdaLR):
    # https://arxiv.org/pdf/1704.00109.pdf
    def __init__(self, optimizer: torch.optim.Optimizer, training_iterations, n_models):
        self.iter_per_model = training_iterations // n_models

        T_over_M = math.ceil(training_iterations / n_models)
        lr_lambda = lambda iteration: 0.5 * (  # noqa: E731
                torch.cos(torch.tensor(math.pi * (iteration % T_over_M) / T_over_M)) + 1
        )

        super(SnapshotEnsembleLRScheduler, self).__init__(optimizer=optimizer, lr_lambda=lr_lambda)


@dataclass
class Snapshot:
    state_dict: dict
    loss: float


def ensemble(device,
             model: BaseModule,
             train_loader: DataLoader,
             val_loader: DataLoader,
             optimizer: torch.optim.Optimizer,
             criterion: torch.nn.Module,
             lr_scheduler: SnapshotEnsembleLRScheduler,
             epochs: int,
             log_writer: SummaryWriter
             ):
    snapshots = []

    train_batches = len(train_loader)

    iter_per_model = lr_scheduler.iter_per_model
    global_step = 0

    total_train_loss = 0
    total_train_samples = 0

    curr_index = 0

    for epoch in range(epochs):
        for batch_index, (X, y) in enumerate(tqdm(train_loader, total=train_batches, unit="batch", leave=False)):
            model.train()

            optimizer.zero_grad()

            batch_size = get_batch_size(X)
            total_train_samples += batch_size

            X, y = move_to_device(X, y, device)

            out = forward(model, X)

            loss = criterion(out, y)
            loss.backward()
            total_train_loss += loss.item() * batch_size

            optimizer.step()
            lr_scheduler.step()
            global_step += 1

            if (global_step % iter_per_model) == 0:
                # snapshot
                val_loss = validation(device, model, val_loader, criterion)

                snapshot = copy.deepcopy(model)
                snapshots.append(Snapshot(state_dict=snapshot.state_dict(), loss=val_loss))

        total_train_loss = total_train_loss / total_train_samples

        print(f"Epoch {epoch}/{epochs}")
        print(f"Train loss model {curr_index}: {total_train_loss}")
        if (global_step % iter_per_model) == 0:
            print(f"Validation loss model {curr_index}: {val_loss}")

        log_writer.add_scalar("ensemble/lr", optimizer.param_groups[0]['lr'], epoch)
        log_writer.add_scalar("ensemble/train/loss", total_train_loss, epoch)

        if (global_step % iter_per_model) == 0:
            log_writer.add_scalar("ensemble/val/loss", val_loss, epoch)

        if (global_step % iter_per_model) == 0:
            # reset
            curr_index += 1
            total_train_loss = 0
            total_train_samples = 0

    return snapshots


@torch.no_grad()
def validation(device, model: BaseModule, loader: DataLoader, criterion: torch.nn.Module):
    model.eval()
    total_loss = 0
    total_samples = 0
    for batch_index, (X, y) in enumerate(tqdm(loader, total=len(loader), unit='batch', leave=False)):
        batch_size = get_batch_size(X)
        total_samples += batch_size

        X, y = move_to_device(X, y, device)
        out = forward(model, X)
        # non averaged loss
        total_loss += criterion(out, y).item() * batch_size

    return total_loss / total_samples


def create_average_snapshot(snapshots: List[Snapshot]):
    # create snapshot model
    # average parameters
    total_snapshots = len(snapshots)
    iter_snapshots = iter(snapshots)

    average_snapshot = next(iter_snapshots).state_dict

    for snapshot in iter_snapshots:
        for key in average_snapshot:
            average_snapshot[key] += snapshot.state_dict[key]

    for key in average_snapshot:
        average_snapshot[key] = average_snapshot[key] / total_snapshots

    return average_snapshot


if __name__ == '__main__':
    fix_randomness(2020)

    model_kwargs = {
        "n_dab": 8,
        "n_intra_layers": 4,
        "out_channels_dab": 32,
        "intra_layer_output_features": 32
    }
    patch_size = 48
    num_workers = 4
    batch_size = 8
    pin_memory = True
    epochs = 500

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. Cannot train the network.")
    device = torch.device('cuda:0')
    lfr = LaplacianFrequencyRepresentation(1, 2, 11)
    model = ASDN(input_image_channels=3, lfr=lfr, **model_kwargs).to(device)
    asdn_collate_fn = partial(create_batch_for_training, lfr=lfr)

    # Default dataset parameter
    default_dataset_params = dict(patch_size=patch_size, lfr=lfr)
    default_dataloader_params = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=asdn_collate_fn,
                                     pin_memory=pin_memory)

    # Prepare datasets
    augmentation = RandomCompose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        Rotate([-90, 90], expand=True)
    ])

    DATASET_DIR = get_datasets_dir() / "DIV2K"
    TRAIN_DATASET = ASDNDataset(DATASET_DIR / "DIV2K_train_HR", augmentation=augmentation, **default_dataset_params)
    VAL_DATASET = ASDNDataset(DATASET_DIR / "DIV2K_valid_HR", augmentation=None, **default_dataset_params)

    TRAIN_DATALOADER = FastDataLoader(dataset=TRAIN_DATASET, shuffle=True, **default_dataloader_params)
    VAL_DATALOADER = FastDataLoader(dataset=VAL_DATASET, shuffle=False, **default_dataloader_params)

    initial_lr = 0.001
    adam = Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-8)

    lr_scheduler = SnapshotEnsembleLRScheduler(optimizer=adam, training_iterations=epochs * len(
        TRAIN_DATALOADER), n_models=5)

    # Define the criterion: L1 mean over the batch
    l1 = L1Loss(reduction='mean').to(device)

    path = get_models_dir() / model.__class__.__name__ / "ensemble"
    if not path.exists():
        path.mkdir(parents=True)
    log_writer = SummaryWriter(str(path))

    snapshots_path = path / "snapshots"
    if not snapshots_path.exists():
        snapshots_path.mkdir(parents=True)
    snapshots_list_path = snapshots_path / "snapshots.list"

    if snapshots_list_path.exists():
        print("Loading previous snapshots...")
        snapshots = torch.load(snapshots_list_path)
    else:
        snapshots = ensemble(device, model, TRAIN_DATALOADER, VAL_DATALOADER, adam, l1, lr_scheduler, epochs,
                             log_writer)
        torch.save(snapshots, snapshots_list_path)

    # state_dict
    average_model = create_average_snapshot(snapshots)

    torch.save(average_model, snapshots_path / "average_model.pytorch")
