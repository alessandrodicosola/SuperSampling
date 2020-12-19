import torch
from torch.utils.data import Dataset

from base import BaseModule


class RandomDataset(Dataset):
    def __init__(self, n_input, input_size, target_size, size_dataset=8):
        self.size_dataset = size_dataset
        self.n_input = n_input
        self.input_size = input_size
        self.target_size = target_size

    def __len__(self):
        return self.size_dataset

    def __getitem__(self, item):
        # (X,y)
        return tuple([torch.rand(self.input_size) for _ in range(self.n_input)]), torch.rand(self.target_size)


class NetworkOneInput(BaseModule):

    def train_step(self, input):
        return self(input)

    @torch.no_grad()
    def val_step(self, input):
        return self(input)

    @torch.no_grad()
    def test_step(self, input):
        return self(input)

    def __init__(self, input_size, low_features):
        super(NetworkOneInput, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=input_size[0], out_channels=low_features, kernel_size=3, padding=1)
        self.out = torch.nn.Conv2d(in_channels=low_features, out_channels=input_size[0], kernel_size=3, padding=1)

    def forward(self, input) -> torch.Tensor:
        return self.out(self.conv(input))


class NetworkTwoInputs(BaseModule):

    def __init__(self, input_size, low_features):
        super(NetworkTwoInputs, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=input_size[0], out_channels=low_features, kernel_size=3, padding=1)
        self.out = torch.nn.Conv2d(in_channels=low_features, out_channels=input_size[0], kernel_size=3, padding=1)

    def forward(self, input1, input2) -> torch.Tensor:
        out1 = self.conv(input1)
        out2 = self.conv(input2)
        out = out1 + out2
        return self.out(out)

    def train_step(self, input1, input2):
        return self(input1, input2)

    @torch.no_grad()
    def val_step(self, input1, input2):
        return self(input1, input2)

    @torch.no_grad()
    def test_step(self, input1, input2):
        return self(input1, input2)

class BaseModuleTwoOutput(BaseModule):
    def forward(self, input):
        # do some computation otherwise requires_grad is not set
        input = input + 0.0
        return (input, input)

    def train_step(self, input):
        return self(input)

    @torch.no_grad()
    def val_step(self, input):
        return self(input)

    @torch.no_grad()
    def test_step(self, input):
        return self(input)
