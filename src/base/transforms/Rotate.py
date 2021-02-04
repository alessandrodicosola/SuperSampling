import random
from typing import List

import torch
from torchvision.transforms import functional as F


class Rotate(torch.nn.Module):
    """Apply a single rotation sampled from a list
    Args:
        degrees: list of int degrees
        kwargs: optional argument of torchvision.transforms.functional.rotate
    Keyword Args:
        #TODO: Write args
    """

    def __init__(self, degrees: List, **kwargs):
        super(Rotate, self).__init__()
        self.degrees = degrees
        self.rotate_transform_kwargs = kwargs

    def forward(self, input):
        degree = random.choice(self.degrees)
        return F.rotate(input, degree, **self.rotate_transform_kwargs)
