import unittest
from functools import partial
from unittest import TestCase
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from base import BaseModel
from base.Trainer import Trainer
from datasets.ASDNDataset import ASDNDataset, collate_fn
from models.ASDN import ASDN
from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation


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


class NetworkTwoInputs(BaseModel):

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


class TestTrainer(TestCase):

    @unittest.skip("EXPENSIVE")
    def test_fit_with_ASDN(self):
        lfr = LaplacianFrequencyRepresentation(1, 2, 11)
        collate_fn_lfr = partial(collate_fn, lfr=lfr)

        val_dataset = ASDNDataset("DIV2K_valid_HR", 24, lfr)
        val_loader = DataLoader(val_dataset, 64, False, num_workers=4, collate_fn=collate_fn_lfr)
        train_dataset = ASDNDataset("DIV2K_train_HR", 24, lfr)
        train_loader = DataLoader(train_dataset, 32, True, num_workers=4, collate_fn=collate_fn_lfr)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        asdn = ASDN(3, lfr, n_dab=3, n_intra_layers=3, out_compressed_channels=32, out_channels_dab=8,
                    intra_layer_output_features=8).to(device)

        adam = Adam(asdn.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        trainer = Trainer("test", asdn, adam, MSELoss().to(device), device=device)
        trainer.fit(train_loader, val_loader, 1)

    def test_fit_one_input(self):
        device = torch.device('cpu')
        input_size = (3, 24, 24)
        dataset = RandomDataset(1, input_size, input_size)
        dataloader = DataLoader(dataset, batch_size=32)
        loss = MSELoss().to(device)
        model = NetworkOneInput(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        trainer = Trainer("test", model, adam, loss, device=device)
        trainer.fit(dataloader, dataloader, 5)

    def test_fit_two_inputs(self):
        def collate_fn(batch):
            inputs, target = zip(*batch)
            input1, input2 = zip(*inputs)
            return (torch.stack(input1, dim=0), torch.stack(input2, dim=0)), torch.stack(target, dim=0)

        device = torch.device('cpu')
        input_size = (3, 24, 24)
        dataset = RandomDataset(2, input_size, input_size)
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        loss = MSELoss().to(device)
        model = NetworkTwoInputs(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        trainer = Trainer("test", model, adam, loss, device=device)
        trainer.fit(dataloader, dataloader, 5)


if __name__ == "__main__":
    unittest.main()
