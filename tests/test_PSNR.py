import unittest
from unittest import TestCase

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from base import BaseModule
from base.Trainer import Trainer
from base.metrics.PSNR import PSNR
from tests.pytorch_test import PyTorchTest
from tests.util_for_testing import RandomDataset, NetworkOneInput


class TestPSNR(PyTorchTest):
    def before(self):
        device = torch.device("cpu")
        input_size = (3, 24, 24)
        dataset = RandomDataset(1, input_size, input_size)
        self.dataloader = DataLoader(dataset, batch_size=32)
        loss = MSELoss().to(device)
        model = NetworkOneInput(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        self.trainer = Trainer("test", model, adam, loss, metric=PSNR(), device=device)

    def after(self):
        self.dataloader = None
        self.trainer = None
        del self.trainer

    def test_forward(self):
        try:
            self.trainer.fit(self.dataloader, self.dataloader, 5)
        except:
            self.fail("Error during forward")

    def test_same(self):
        i1 = torch.rand(32, 3, 48, 48)
        psnr = PSNR(max_pixel_value=1.)(i1, i1)
        self.assertAlmostEqual(psnr, 100)

    def test_raise_error(self):
        i1 = torch.rand(3, 5, 5)
        i2 = torch.rand(3, 5, 12)
        psnr = PSNR(max_pixel_value=1.)
        self.assertRaises(RuntimeError, psnr, i1, i2)


if __name__ == "__main__":
    unittest.main()
