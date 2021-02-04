from pathlib import Path

import PIL
import torch
import torchvision


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

