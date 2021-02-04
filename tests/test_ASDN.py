import math
from functools import reduce
from unittest import TestCase, skip

from torch.utils.data import DataLoader

from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation
from models.ASDN import ASDN

import torch
import torch.nn as nn

from tests.pytorch_test import PyTorchTest
from tests.util_for_testing import RandomTensorDataset


@skip("Done and it's working. Skipped because it's expensive.")
class TestASDN(PyTorchTest):
    def before(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Current device: ", self.device)

        # batch size 8 use 5.5GB
        self.loader = DataLoader(RandomTensorDataset(), batch_size=8)
        self.asdn = ASDN(3, lfr=LaplacianFrequencyRepresentation(1, 2, 11)).to(self.device)

    def after(self):
        self.device = None
        self.loader = None
        self.asdn = None

    def test_forward(self):
        scale = 1.3674

        leveliminus1, leveli = self.lfr.get_for(scale=scale)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        self.asdn.train()

        for index, batch in enumerate(self.loader):
            batch = batch.to(self.device)

            start.record()
            outputi = self.asdn(batch, irb_index=leveli.index)
            end.record()

            torch.cuda.synchronize()
            print("=" * 3, index, "=" * 3)
            print(f"Time passed (s): {start.elapsed_time(end) / 1000:.2f}")

            print(outputi.size())


class TestASDN(TestCase):

    def test_test_step(self):
        def compute_last_patch_size(scale, patch_size):
            scales = lfr.get_scales(scale)
            scales = [lfr.get_for(scale)[-1].scale for scale in scales]
            last_patch_size = patch_size
            for scale in scales:
                last_patch_size = last_patch_size * scale
            return math.floor(last_patch_size)

        lfr = LaplacianFrequencyRepresentation(1, 2, 11)
        model = ASDN(3, lfr).cuda()

        patch = torch.rand(8, 3, 48, 48).cuda()
        print(patch.size())
        print("x2", patch.size(-1) * 2)
        print("x3.5", patch.size(-1) * 3.5)

        scale = 2
        out = model.test_step(scale, patch)
        self.assertEqual(out.size(-1), compute_last_patch_size(scale, patch.size(-1)))

        scale = 3.5
        out = model.test_step(scale, patch)
        self.assertEqual(out.size(-1), compute_last_patch_size(scale, patch.size(-1)))
