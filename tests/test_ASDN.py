from unittest import TestCase

from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation
from models.ASDN import ASDN

import torch
import torch.nn as nn




class TestASDN(TestCase):
    def test_forward(self):
        from torch.utils.data import DataLoader, Dataset

        cpu = "cpu"
        gpu = "cuda:0"

        scale = 1.3674
        lfr = LaplacianFrequencyRepresentation(1, 2, 11)
        leveliminus1, leveli = lfr.get_for(scale=scale)

        class RandomTensorDataset:
            def __init__(self):
                self.patch = (3, 48, 48)
                self.len = 64

            def __getitem__(self, index):
                return torch.rand(*self.patch)

            def __len__(self):
                return self.len

        # batch size 8 use 5.5GB
        loader = DataLoader(RandomTensorDataset(), batch_size=8)

        asdn = ASDN(3,lfr=lfr).to(gpu)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        asdn.train()

        for index, (scale), (low_res_batch_i_minus_1, pyramid_i_minus_1), (low_res_batch_i, pyramid_i) in enumerate(loader):
            start.record()
            outputi = asdn(low_res_batch_i, irb_index=leveli.index).to(cpu)
            outputiminus1 = asdn(low_res_batch_i_minus_1, irb_index=leveliminus1.index).to(cpu)
            end.record()

            print("=" * 3, index, "=" * 3)
            print(f"Time passed (s): {start.elapsed_time(end) / 1000:.2f}")

            print(outputi.size())
            print(outputiminus1.size())

        # !nvidia-smi
