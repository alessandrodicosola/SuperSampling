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


class TestASDNASDNDataset(TestCase):
    def test_together(self):
        torch.cuda.empty_cache()

        cpu = "cpu"
        gpu = "cuda:0"

        lfr = LaplacianFrequencyRepresentation(1, 2, 11)

        PATCH_SIZE = 48

        # no more no less
        BATCH_SIZE = 8

        COLLATE_FN = partial(collate_fn, lfr=lfr)
        dataset = ASDNDataset(R"DIV2K_valid_HR", PATCH_SIZE, lfr)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=6, pin_memory=True, collate_fn=COLLATE_FN)
        asdn = ASDN(3, lfr).to(gpu)

        denormalize = NormalizeInverse(dataset.mean, dataset.std)

        for index, ((scale, low_res_batch_i_minus_1, low_res_batch_i), pyramid_i) in enumerate(
                tqdm(dataloader)):

            level_i_minus_1, level_i = lfr.get_for(scale)
            out_size = math.floor(level_i.scale * PATCH_SIZE)

            if index == 0:
                outiminus1 = asdn(low_res_batch_i_minus_1.to(gpu), level_i_minus_1.index)
                outi = asdn(low_res_batch_i.to(gpu), level_i.index)

                assert outi.size(2) == out_size and outi.size(3) == out_size

                interpolated_outiminus1 = interpolating_fn(outiminus1, size=(out_size, out_size))

                phase = interpolated_outiminus1 - outi

                reconstruted = outi + lfr.get_weight(scale) * phase
                reconstruted = reconstruted.cpu().detach()
                reconstruted = denormalize(reconstruted)

                plt.imshow(make_grid(reconstruted[:4], nrow=2).permute(1, 2, 0))
                plt.show()
            else:
                break


if __name__ == "__main__":
    unittest.main()
