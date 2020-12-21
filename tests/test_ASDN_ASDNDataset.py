import math
from functools import partial
from unittest import TestCase, skip
import unittest

# test ASDN and Dataset together
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from datasets.ASDNDataset import ASDNDataset, collate_fn, interpolating_fn, NormalizeInverse
from models.ASDN import ASDN
from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation

import matplotlib.pyplot as plt

from tests.pytorch_test import PyTorchTest

@skip("Done and it's working. Skipped because it's expensive.")
class TestASDNASDNDataset(PyTorchTest):
    def before(self):
        self.cpu = "cpu"
        self.gpu = "cuda:0"

        self.lfr = LaplacianFrequencyRepresentation(1, 2, 11)

        self.PATCH_SIZE = 48

        # no more no less
        BATCH_SIZE = 8

        COLLATE_FN = partial(collate_fn, lfr=self.lfr)
        dataset = ASDNDataset(R"DIV2K_valid_HR", self.PATCH_SIZE, self.lfr)
        self.denormalize = NormalizeInverse(dataset.mean, dataset.std)
        self.dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=6, pin_memory=True,
                                     collate_fn=COLLATE_FN)
        self.asdn = ASDN(3, self.lfr).to(self.gpu)

    def after(self):
        self.cpu = None
        self.gpu = None
        self.dataloader = None
        self.asdn = None

    def test_together(self):

        for index, ((scale, low_res_batch_i_minus_1, low_res_batch_i), pyramid_i) in enumerate(
                tqdm(self.dataloader)):

            level_i_minus_1, level_i = self.lfr.get_for(scale)
            out_size = math.floor(level_i.scale * self.PATCH_SIZE)

            if index == 0:
                outiminus1 = self.asdn(low_res_batch_i_minus_1.to(self.gpu), level_i_minus_1.index)
                outi = self.asdn(low_res_batch_i.to(self.gpu), level_i.index)

                assert outi.size(2) == out_size and outi.size(3) == out_size

                interpolated_outiminus1 = interpolating_fn(outiminus1, size=(out_size, out_size))

                phase = interpolated_outiminus1 - outi

                reconstruted = outi + self.lfr.get_weight(scale) * phase
                reconstruted = reconstruted.cpu().detach()
                reconstruted = self.denormalize(reconstruted)

                plt.imshow(make_grid(reconstruted[:4], nrow=2).permute(1, 2, 0))
                plt.show()
            else:
                break


if __name__ == "__main__":
    unittest.main()
