import random
from typing import List

import torch


class RandomCompose(torch.nn.Module):
    """ Apply a sequence of transformation randomly sampled from a list

    Args:
        transforms: list of transformation
        weights: weight for each transformation
    """

    def __init__(self, transforms: List, weights: List = None):
        super(RandomCompose, self).__init__()
        self.transforms = transforms
        self.weights = weights
        self.single_probability = 1 / len(transforms)

    def forward(self, input):
        output = input
        transformations = random.choices(self.transforms, weights=self.weights)
        for transform in transformations:
            output = transform(output)
        return output
