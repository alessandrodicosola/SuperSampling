import math
from pathlib import Path
import random
from typing import Union, List

import PIL.Image as Image
import torchvision
from torch.utils.data import Dataset

from datasets.ASDNDataset import ASDNDataset
import torchvision.transforms.functional as F


class TestDataset(Dataset):
    def __init__(self, testing_scales: List[float], path: Union[str, Path], format: str = "png"):
        super(TestDataset, self).__init__()

        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise RuntimeError(f"Path not found: {path}")

        self.scales = testing_scales
        self.files = list(path.glob(f"*.{format}"))
        self.to_tensor = torchvision.transforms.ToTensor()

        self.normalize = ASDNDataset.normalize_fn()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # select a random scale from ones specified
        scale = random.choice(self.scales)

        image_file = self.files[index]
        with Image.open(image_file) as image:
            ### LAZY SOLUTION FOR GRAY SCALE IMAGES
            bands = image.getbands()
            if len(bands) == 1:
                image = image.convert("RGB")
            ###

            # is an inplace function
            # transform image into a squared image for simplifying the process of testing
            thumb_size = [min(image.size)] * 2
            new_size = [math.ceil(thumb_size[0] / scale)] * 2

            ground_truth = self.to_tensor(F.resize(image, thumb_size))
            lr_image = self.normalize(F.resize(ground_truth, new_size))

        return scale, lr_image, ground_truth
