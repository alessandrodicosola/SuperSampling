# python file where train and test command will be run
import random
from functools import partial

import torch
import numpy as np
import torchvision
from torch.nn import L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from base.callbacks.TensorboardCallback import TensorboardCallback
from base.transforms.RandomCompose import RandomCompose
from base.transforms.Rotate import Rotate
from base.trainer.Trainer import Trainer
from base.metrics.PSNR import PSNR
from base.metrics.SSIM import SSIM
from datasets.ASDNDataset import ASDNDataset, create_batch_for_training
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


def run(experiment: str,
        training_kwargs: dict,
        other_kwargs: dict,
        model_kwargs: dict):
    """
    Run training

    Args:
        experiment:
        save:
        training_kwargs:
        model_kwargs:
        other_kwargs:

    Keyword Args:
        # training_kwargs

        epochs (mandatory)

        batch_size (mandatory)

        save_checkpoints (default 1)

        lfr_min (default 1)

        lfr_max (default 2)

        lfr_count (default 11)

        patch_size (default 48)

        augmentation (default None)

        lr (default 3e-4)

        lr_scheduler (default None)

        #other_kwargs

        num_workers (default 2)

        pin_memory (default True)

        filter_out (default None. List of keys to avoid in experiment string taken from training_kwargs and model_kwargs)

        save (default False. Save state_dict model)

        #model_kwargs

        see ASDN kwargs

    Returns:
        error (bool): True if there was an error, False otherwise
    """

    def get_repr(obj):
        import re

        obj_str = str(obj)
        pattern = r"<class \'[\w\.]+?(?P<name>\w+)\'>,\s(?P<args>[^)]+)"
        match = re.search(pattern, obj_str)
        if match is not None:
            # res32_lr_schedulerfunctools.partial(<class 'torch.optim.lr_scheduler.StepLR'>, step_size=20, gamma=0.1)
            name = match.group('name').strip()
            args = "_".join(map(lambda t: t.strip(), match.group('args').strip().split(',')))
            return name + "_" + args
        return obj_str

    ##############################################################
    # extract training kwargs
    epochs = training_kwargs.pop("epochs")
    batch_size = training_kwargs.pop("batch_size")
    save_checkpoints = training_kwargs.pop("save_checkpoint", 1)
    lfr_min = training_kwargs.pop("start_lfr", 1)
    lfr_max = training_kwargs.pop("end_lfr", 2)
    lfr_count = training_kwargs.pop("count_lfr", 11)
    patch_size = training_kwargs.pop("patch_size", 48)
    augmentation = training_kwargs.pop("augmentation", None)
    lr = training_kwargs.pop("lr", 3e-4)
    lr_scheduler = training_kwargs.pop("lr_scheduler", None)

    # extract other_kwargs
    num_workers = other_kwargs.pop("num_workers", 2)
    pin_memory = other_kwargs.pop("pin_memory", True)
    filter_out = other_kwargs.pop("filter_out", set()) | set("save_checkpoint")
    save = other_kwargs.pop("save", False)
    ##############################################################

    assert isinstance(filter_out, set), "filter_out must be a list"


    # set save checkpoints variables
    models.ASDN.set_save_checkpoints(save_checkpoints)

    # Set experiment string
    experiment += f"_E{epochs}_B{batch_size}_S{models.ASDN._SEGMENTS_GRADIENT_CHECKPOINT}"

    kwargs_to_print = {**training_kwargs, **model_kwargs}
    if len(kwargs_to_print) > 0:
        items_to_print = [(key, item) for key, item in kwargs_to_print.items() if key not in filter_out]
        experiment += "_" + "_".join(map(lambda elem: f"{elem[0]}{get_repr(elem[1])}", items_to_print))
    #Print
    print(f"Starting {experiment}")
    # Get the device
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. Cannot train the network.")
    device = torch.device('cuda:0')

    # Get the laplacian frequency representation between 1 and 2
    lfr = LaplacianFrequencyRepresentation(lfr_min, lfr_max, lfr_count)

    # Define the collate function
    asdn_collate_fn = partial(create_batch_for_training, lfr=lfr)

    # Default dataset parameter
    default_dataset_params = dict(patch_size=patch_size, lfr=lfr)
    default_dataloader_params = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=asdn_collate_fn,
                                     pin_memory=pin_memory)

    # Prepare datasets

    DATASET_DIR = get_datasets_dir() / "DIV2K"
    TRAIN_DATASET = ASDNDataset(DATASET_DIR / "DIV2K_train_HR", augmentation=augmentation, **default_dataset_params)
    VAL_DATASET = ASDNDataset(DATASET_DIR / "DIV2K_valid_HR", augmentation=None, **default_dataset_params)

    TRAIN_DATALOADER = FastDataLoader(dataset=TRAIN_DATASET, shuffle=True, **default_dataloader_params)
    VAL_DATALOADER = FastDataLoader(dataset=VAL_DATASET, shuffle=False, **default_dataloader_params)

    # Prepare the model
    # Prepare model with specified parameters
    ASDN_ = models.ASDN.ASDN(input_image_channels=3, lfr=lfr, **model_kwargs).to(device)

    # lr=3e-4 as suggested in https://karpathy.github.io/2019/04/25/recipe/

    ADAM = Adam(ASDN_.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    LR_SCHEDULER = lr_scheduler(optimizer=ADAM) if lr_scheduler else None

    # Define the criterion: L1 mean over the batch
    L1 = L1Loss(reduction='mean').to(device)

    # Define the metrics
    METRICS = [PSNR(dynamic_range=1.0, reduction='mean', denormalize_fn=ASDNDataset.denormalize_fn()),
               SSIM(dynamic_range=1.0, reduction='mean', denormalize_fn=ASDNDataset.denormalize_fn())]

    # Define the trainer
    TRAINER = Trainer(experiment=experiment, model=ASDN_, optimizer=ADAM, criterion=L1, metric=METRICS,
                      device=device,
                      lr_scheduler=LR_SCHEDULER,
                      callback=None)

    tensorboard_callback = TensorboardCallback(log_dir=str(TRAINER.log_dir), print_images=True,
                                               print_images_frequency=10,
                                               denormalize_fn=ASDNDataset.denormalize_fn())

    TRAINER.callback.add_callback(tensorboard_callback)

    error: bool = False
    try:
        TRAINER.fit(train_loader=TRAIN_DATALOADER, val_loader=VAL_DATALOADER, epochs=epochs)
        if save:
            TRAINER.save_model(f"_last")

    except (RuntimeError, ValueError) as e:
        print(str(e))
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


def batch_time_default():
    fix_randomness(2020)
    other_kwargs = dict(n_workers=0, pin_memory=False)
    training_kwargs = dict(save_checkpoints=2, epochs=1, batch_size=4)
    model_kwargs = {}
    run("BATCH_TIME", other_kwargs=other_kwargs, training_kwargs=training_kwargs, model_kwargs=model_kwargs)


def batch_time_comparison():
    fix_randomness(2020)

    model_kwargs = {
        "n_dab": 5,
        "n_intra_layers": 5,
        "out_channels_dab": 32,
        "intra_layer_output_features": 40
    }
    other_kwargs = dict(n_workers=2, pin_memory=True)
    training_kwargs = dict(save_checkpoints=1, epochs=1, batch_size=8)

    # no save checkpoint
    run("BATCH_TIME", other_kwargs=other_kwargs, training_kwargs=training_kwargs, model_kwargs=model_kwargs)

    # save checkpoint
    training_kwargs['save_checkpoints'] = 2
    run("BATCH_TIME", other_kwargs=other_kwargs, training_kwargs=training_kwargs, model_kwargs=model_kwargs)


def overfitting(epochs: int, **model_kwargs):
    fix_randomness(2020)

    others_params = {
        "lr": 1e-3
    }

    kwargs = {**model_kwargs, **others_params}

    n_workers = 2 if kwargs.get("out_channels_dab") >= 32 else 4
    other_kwargs = dict(n_workers=n_workers, pin_memory=True)
    training_kwargs = dict(save_checkpoints=1, epochs=epochs, batch_size=8)

    run("OVERFITTING",
        other_kwargs=other_kwargs,
        training_kwargs=training_kwargs,
        model_kwargs=model_kwargs)


def show_summary(batch_size=8, save_checkpoints=1, **model_kwargs):
    from torchsummary import summary
    from models.ASDN import ASDN

    input_shape = (3, 48, 48)
    interpolated_shape = (3, 96, 96)
    irb_index = 10

    LFR = LaplacianFrequencyRepresentation(1, 2, 11)
    if save_checkpoints > 1:
        models.ASDN.set_save_checkpoints(save_checkpoints)

    model = models.ASDN.ASDN(input_image_channels=3, lfr=LFR, **model_kwargs).cuda()
    model.forward = partial(model.forward, irb_index=irb_index)
    return summary(model, input_size=interpolated_shape, batch_size=batch_size)


def train(epochs: int, **model_kwargs):
    fix_randomness(2020)

    n_workers = 2 if model_kwargs.get("out_channels_dab") >= 32 else 4
    other_kwargs = dict(n_workers=n_workers, pin_memory=True, save=True)
    training_kwargs = dict(save_checkpoints=1, epochs=epochs, batch_size=8, lr=0.001,
                           lr_scheduler=partial(ReduceLROnPlateau, mode="min", factor=0.5, patience=10, verbose=True))

    run("TRAINING", other_kwargs=other_kwargs, training_kwargs=training_kwargs, model_kwargs=model_kwargs)


def train_with_augmentation(epochs: int, **model_kwargs):
    fix_randomness(2020)

    n_workers = 2 if model_kwargs.get("out_channels_dab") >= 32 else 4

    training_kwargs = dict(save_checkpoints=1, epochs=epochs, batch_size=8, lr=0.001,
                           lr_scheduler=partial(ReduceLROnPlateau, mode="min", factor=0.5, patience=10, verbose=True),
                           augmentation=RandomCompose([
                               torchvision.transforms.RandomHorizontalFlip(p=1.0),
                               torchvision.transforms.RandomVerticalFlip(p=1.0),
                               Rotate([-90, 90], expand=True)
                           ]),
                           )

    n_workers = 2 if model_kwargs.get("out_channels_dab") >= 32 else 4
    other_kwargs = dict(n_workers=n_workers, pin_memory=True, save=True)
    run("TRAINING_AUGMENTATION", other_kwargs=other_kwargs, training_kwargs=training_kwargs, model_kwargs=model_kwargs)

if __name__ == "__main__":
    # default params for the project
    model_kwargs = {
        "n_dab": 8,
        "n_intra_layers": 4,
        "out_channels_dab": 32,
        "intra_layer_output_features": 32
    }
    # show_summary(**model_kwargs)
    # # 3h48m of training
    # model_kwargs = {
    #     "n_dab": 6,
    #     "n_intra_layers": 5,
    #     "out_channels_dab": 50,
    #     "intra_layer_output_features": 32
    # }
    # show_summary(**model_kwargs)

    # overfitting(epochs=200, **model_kwargs)
    # train(500, **model_kwargs)
    pass
