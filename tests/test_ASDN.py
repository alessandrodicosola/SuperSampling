from unittest import TestCase, skip

from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation
from models.ASDN import ASDN

import torch
import torch.nn as nn




class TestASDN(TestCase):
    def test_forward(self):
        from torch.utils.data import DataLoader, Dataset

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Current device: ", device)


        scale = 1.3674
        lfr = LaplacianFrequencyRepresentation(1, 2, 11)
        leveliminus1, leveli = lfr.get_for(scale=scale)

        class RandomTensorDataset(Dataset):
            def __init__(self):
                self.patch = (3, 48, 48)
                self.len = 16

            def __getitem__(self, index):
                return torch.rand(*self.patch)

            def __len__(self):
                return self.len

        # batch size 8 use 5.5GB
        loader = DataLoader(RandomTensorDataset(), batch_size=8)



        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        asdn = ASDN(3, lfr=lfr).to(device)

        asdn.train()

        for index, batch in enumerate(loader):

            batch = batch.to(device)

            start.record()
            outputi = asdn(batch, irb_index=leveli.index)
            end.record()

            torch.cuda.synchronize()
            print("=" * 3, index, "=" * 3)
            print(f"Time passed (s): {start.elapsed_time(end) / 1000:.2f}")

            print(outputi.size())