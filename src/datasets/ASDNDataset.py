import math
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn.functional
import torch.utils.data
import torchvision.transforms

from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation

import pickle

import PIL

interpolating_fn = partial(torch.nn.functional.interpolate,
                           mode="bicubic",
                           align_corners=False,
                           recompute_scale_factor=False)


# put collate_fn at top level in order to be pickable and used inside DataLoader
def collate_fn(batch, lfr: LaplacianFrequencyRepresentation):
    """Function for generating a batch

    Args:
        batch: input batch
        lfr:
        interpolating_fn: torch.nn.functional.interpolate

    Returns:
        Tuple: (X,y)
            (scale, low_res_batch_i_minus_1, pyramid_i_minus_1),
            (low_res_batch_i, pyramid_i)
    """

    # sample a random value between 1 and 2 using an uniform distribution
    # this will be the scale value used for the batch
    scale = random.uniform(lfr.start, lfr.end)
    level_i_minus_1, level_i = lfr.get_for(scale)

    input, pyramid_batch = zip(*batch)
    # create the input batch
    low_res_batch = torch.stack(input)

    # interpolated the LR input for forwarding correctly inside the network
    # interpolating bicubic cause overshot => clamp between min and max

    low_res_batch_i = interpolating_fn(low_res_batch, scale_factor=level_i.scale)

    if level_i_minus_1.index == 0:
        # found bug:  level i-1 contains empty images interpolating using scale 1.xxxxx as factor
        low_res_batch_i_minus_1 = low_res_batch
    else:
        low_res_batch_i_minus_1 = interpolating_fn(low_res_batch, scale_factor=level_i_minus_1.scale)

    # create the ground truth for the loss
    pyramid_i = []

    for elem in pyramid_batch:
        assert isinstance(elem, list)
        assert len(elem) == lfr.count

        laplacian_i = elem[level_i.index]
        pyramid_i.append(laplacian_i)

    pyramid_i = torch.stack(pyramid_i)

    # return (X,y)
    return (scale, low_res_batch_i_minus_1, low_res_batch_i), pyramid_i


def compute_mean_std(path: str) -> Dict[str, List[float]]:
    """Compute the mean and std of the dataset inside the folder `path`

    Args:
        path: absolute or relative path

    Returns:
        Dictionary:
            {
                mean : List[float]
                std  : List[float]
            }
    """

    from pathlib import Path
    import PIL.Image as PIL
    from tqdm.auto import tqdm
    path = Path(path)

    class ImageDataset(torch.utils.data.Dataset):
        """Simple Dataset that returns image from a folder

        Args:
            path: relative or absolute path of the folder that contains images
            format: format of the images (Default: png)
        """

        def __init__(self, path: str, format: str = "png"):
            path = Path(path)
            self.files = list(path.glob(f"*.{format}"))
            self.to_tensor = torchvision.transforms.ToTensor()

        def __len__(self):
            return len(self.files)

        def __getitem__(self, index) -> torch.Tensor:
            with open(self.files[index], "rb") as image_file:
                return self.to_tensor(PIL.open(image_file))

    dataset = ImageDataset(path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=8)

    total: int = dataset.__len__()
    mean: torch.Tensor = 0
    std: torch.Tensor = 0

    for batch in tqdm(loader, total=total, unit="image"):
        image = batch.view(batch.size(0), batch.size(1), -1)
        mean += image.mean(2).sum(0)
        std += image.std(2).sum(0)

    return {"mean": (mean / total).tolist(),
            "std": (std / total).tolist()}


class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, max_pixel_value, mean: List[float], std: List[float]) -> None:
        """Reconstructs the images in the input domain by inverting
        the normalization transformation.

        Args:
            max_pixel_value: 1 if intensities are normalized else 255
            mean: the mean used to normalize the images.
            std: the standard deviation used to normalize the images.
        """
        assert max_pixel_value in [1, 255]
        self.max_pixel_value = max_pixel_value

        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone()).clamp(0, self.max_pixel_value)


class ASDNDataset(torch.utils.data.Dataset):
    """Define the Dataset class used for constructing batches for ASDN network

    NOTE: In the paper is not clear how targets are created therefore this will be my pipeline:
    * 1. Crop a high-res patch with size (patch_size * end, patch_size * end)
    * 2. Create ground truth resizing using the best algorithm
    * 3. Create low-res patch resize using worst algorithm

    Args:
      folder: relative or absolute path or Path object where images are stored
      patch_size: Input patch size (before interpolation )
      lfr: Laplacian Frequency Representation object that stores information of the pyramid
      mean: list of float containing the mean value of each channel
      std: list of float containing the std value of each channel
      mean_std_filename: relative or absolute path of dictionary serialized as a pickle-file that contains mean and std values
      augmentation: Transformations for augmenting the dataset
    """

    @staticmethod
    def denormalize_fn():
        return NormalizeInverse(max_pixel_value=1,
                                mean=[0.4484562277793884, 0.4374960660934448, 0.40452802181243896],
                                std=[0.2436375468969345, 0.23301854729652405, 0.24241816997528076])

    def __init__(self,
                 folder: Union[str, Path],
                 patch_size: int,
                 lfr: LaplacianFrequencyRepresentation,
                 mean: List[float] = None, std: List[float] = None, mean_std_filename: str = None,
                 augmentation=None):

        if isinstance(folder, str):
            folder = Path(folder)

        if not folder.exists:
            raise FileNotFoundError(f"Folder {folder} doesn't exist.")

        if mean_std_filename is not None:
            pickle_file = folder / mean_std_filename
            if not pickle_file.exists():
                raise FileNotFoundError(f"Can't find {pickle_file}")
            else:
                with open(pickle_file, 'rb') as file:
                    dict_mean_std = pickle.load(file)
                    self.mean = dict_mean_std["mean"]
                    self.std = dict_mean_std["std"]
        else:
            if mean is None and std is not None:
                raise RuntimeError("Provide mean.")
            elif mean is not None and std is None:
                raise RuntimeError("Provide std.")
            elif mean is None and std is None:
                # apply my values
                mean_std_dict = {'mean': [0.4484562277793884, 0.4374960660934448, 0.40452802181243896],
                                 'std': [0.2436375468969345, 0.23301854729652405, 0.24241816997528076]}
                self.mean = mean_std_dict['mean']
                self.std = mean_std_dict['std']
            else:
                self.mean = mean
                self.std = std

        self.get_high_res_crop = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            torchvision.transforms.RandomCrop(patch_size * lfr.end),
        ])

        self.resize_functions = [
            torchvision.transforms.Resize(
                (math.floor(level.scale * patch_size), math.floor(level.scale * patch_size)),
                interpolation=PIL.Image.BICUBIC) for level in lfr.information]

        self.resize_high_res_to_low_res = torchvision.transforms.Resize((patch_size, patch_size),
                                                                        interpolation=PIL.Image.BILINEAR)
        self.lfr = lfr

        self.augmentation = augmentation

        self.files = list(folder.glob("*.png"))

    def __getitem__(self, index):
        high_res_image = PIL.Image.open(self.files[index])

        # create high res crop which is the input
        high_res_crop = self.get_high_res_crop(high_res_image)

        # free resources
        high_res_image.close()
        del high_res_image

        # apply augmentation
        if self.augmentation:
            high_res_crop = self.augmentation(high_res_crop)

        # extract LR patch (as input)
        low_res_crop = self.resize_high_res_to_low_res(high_res_crop)

        # extract ground truth
        laplacian_pyramid_ground_truth = [resize_fn(high_res_crop) for resize_fn in self.resize_functions]

        return low_res_crop, laplacian_pyramid_ground_truth

    def __len__(self):
        return len(self.files)
