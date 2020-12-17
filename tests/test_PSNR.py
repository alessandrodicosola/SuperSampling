import unittest
from unittest import TestCase

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from base import BaseModel
from base.Trainer import Trainer
from base.metrics.PSNR import PSNR


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


class NetworkOneInput(BaseModel):

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


class TestPSNR(TestCase):
    def test_forward(self):
        device = torch.device('cpu')
        input_size = (3, 24, 24)
        dataset = RandomDataset(1, input_size, input_size)
        dataloader = DataLoader(dataset, batch_size=32)
        loss = MSELoss().to(device)
        model = NetworkOneInput(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        PSNR_metric = PSNR()
        trainer = Trainer("test", model, adam, loss, metric=PSNR_metric, device=device)
        trainer.fit(dataloader, dataloader, 5)

    def test_same(self):
        i1 = torch.as_tensor([[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]]])
        i2 = torch.as_tensor([[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]]])

        psnr = PSNR(max_pixel_value=1.)(i1, i2)

        self.assertAlmostEqual(psnr, 100)

    def test_raise_error(self):
        i1 = torch.rand(3,5,5)
        i2 = torch.rand(3,5,12)

        psnr = PSNR(max_pixel_value=1.)

        self.assertRaises(RuntimeError, psnr, i1, i2)

if __name__ == "__main__":
        unittest.main()
