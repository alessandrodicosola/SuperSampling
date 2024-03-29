import math
import random
from pathlib import Path

import PIL
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from base.functional import interpolating_fn
from base.metrics.PSNR import PSNR
from base.metrics.SSIM import SSIM
from datasets.ASDNDataset import ASDNDataset
from datasets.TestDataset import TestDataset
from models.ASDN import ASDN
from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation

import torchvision.transforms.functional as F
from torchvision.utils import make_grid

from run_exepriment import fix_randomness
from utility import get_models_dir, get_datasets_dir


@torch.no_grad()
def testing(comment: str, dataset: TestDataset, device, model_state_dict_path: Path, **model_kwargs):
    """Test the model (loaded) on the dataset specified

    Args
        dataset: dataset tot use,
        model_state_dict_path: path of the state dict saved,
        model_kwargs: parameters of the saved model
    """

    if not model_state_dict_path.exists():
        raise FileNotFoundError(model_state_dict_path)

    lfr = LaplacianFrequencyRepresentation(1, 2, 11)

    # Prepare the model
    model = ASDN(input_image_channels=3, lfr=lfr, **model_kwargs)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_state_dict_path))
    assert len(missing_keys) == 0
    assert len(unexpected_keys) == 0
    model = model.to(device)
    model.eval()

    # Prepare the dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    denorm_fn = ASDNDataset.denormalize_fn()

    # Create test logger
    test_folder = f"__TESTING_{dataset.name}_{comment}"
    log_dir = get_models_dir() / model.__class__.__name__ / test_folder
    writer = SummaryWriter(log_dir=str(log_dir))

    psnr_total = 0
    ssim_total = 0
    total = len(dataset)

    ssim_fn = SSIM(reduction="sum", dynamic_range=1.).to(device)
    psnr_fn = PSNR(reduction="sum", dynamic_range=1.).to(device)

    for index, (scale, lr, gt) in enumerate(tqdm(dataloader, total=total, unit='image')):
        global_step = index + 1

        scale = scale.to(device)
        lr = lr.to(device)
        out = model.test_step(scale, lr)

        # sometimes new_size is greater than out due to neighbour levels

        new_width = out.size(-1)
        # resize gt
        gt = F.resize(gt, [new_width] * 2).to(device)

        # .float()/.double() because got errors
        # assert gt.max() <= 1 and gt.min() >= 0
        # img = denorm_fn(out.float())
        # assert img.max() <= 1 and img.min() >= 0

        cur_psnr = psnr_fn(denorm_fn(out.float()), gt.float())
        cur_ssim = ssim_fn(denorm_fn(out.float()), gt.float())

        psnr_total += cur_psnr
        ssim_total += cur_ssim

        nrow = math.ceil(math.sqrt(out.size(0)))

        grid_in = make_grid(denorm_fn(lr), nrow=nrow)
        grid_out = make_grid(denorm_fn(out), nrow=nrow)
        grid_gt = make_grid(gt, nrow=nrow)
        grid_in_bicubic = make_grid(denorm_fn(F.resize(lr, size=new_width, interpolation=PIL.Image.BICUBIC)))

        # scale for each image
        writer.add_scalar(tag="Test/images/scalar", scalar_value=scale, global_step=global_step)
        # psnr current image
        writer.add_scalar(tag="Test/images/psnr", scalar_value=cur_psnr, global_step=global_step)
        # ssim current image
        writer.add_scalar(tag="Test/images/ssim", scalar_value=cur_ssim, global_step=global_step)
        # print test images
        writer.add_image(tag="Test/images/input", img_tensor=grid_in, global_step=global_step)
        writer.add_image(tag="Test/images/output", img_tensor=grid_out, global_step=global_step)
        writer.add_image(tag="Test/images/ground_truth", img_tensor=grid_gt, global_step=global_step)
        writer.add_image(tag="Test/images/bicubic", img_tensor=grid_in_bicubic, global_step=global_step)

    psnr_total = psnr_total / total
    ssim_total = ssim_total / total

    writer.add_text(tag="Test/dataset", text_string=dataset.name)
    writer.add_hparams(hparam_dict=model_kwargs, metric_dict=dict(ssim=ssim_total, psnr=psnr_total))

    writer.close()


if __name__ == "__main__":
    fix_randomness(2020)

    model_kwargs = {
        "n_dab": 8,
        "n_intra_layers": 4,
        "out_channels_dab": 32,
        "intra_layer_output_features": 32
    }

    # Init device
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available. Can't run the testing.")
    device = torch.device("cuda:0")

    # Init datasets
    dataset_names = ["Set5", "Set14", "historical", "Urban100", "Manga109", "BSDS100"]

    aug_train_model = [
        R"D:\University\PROJECTS\MLCV\models\ASDN\TRAINING_AUGMENTATION_E500_B8_S1lr0.001_lr_schedulerReduceLROnPlateau_mode='min'_factor=0.5_patience=10_verbose=True_n_dab8_n_intra_layers4_out_channels_dab32_intra_layer_output_features32\checkpoints\ASDN__last.pytorch"]
    aug_train_model = [Path(path) for path in aug_train_model]
    ensemble_folder = get_models_dir() / "ASDN" / "ensemble" / "snapshots"
    ensemble_models = [path.resolve() for path in ensemble_folder.glob("*.pytorch")]
    models_to_test = aug_train_model + ensemble_models
    #models_to_test = ensemble_models

    comments = ["denorm", "average"]

    for model_state_dict_path, comment in zip(models_to_test, comments):
        print(model_state_dict_path, comment)

        for dataset_name in dataset_names:
            dataset_path = get_datasets_dir() / "Test" / dataset_name
            dataset = TestDataset([3.5, 4], dataset_path)
            dataset.name = dataset_name

            testing(comment, dataset, device, model_state_dict_path, **model_kwargs)
