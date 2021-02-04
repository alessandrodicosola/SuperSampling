from typing import List

import torch
import torchvision.transforms


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