from functools import partial
from typing import Tuple
from unittest import TestCase, skip
import unittest
from torch.utils.data import DataLoader
import torch
from datasets.ASDNDataset import ASDNDataset, collate_fn, NormalizeInverse
from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation
from tests.pytorch_test import PyTorchTest

@skip("Done and it's working. Skipped because it's expensive.")
class TestASDNDataset(PyTorchTest):
    def before(self):
        self.PATCH_SIZE = 48
        self.LFR = LaplacianFrequencyRepresentation(1, 2, 11)
        self.DATASET = ASDNDataset("DIV2K_valid_HR", patch_size=self.PATCH_SIZE, lfr=self.LFR)
        self.COLLATE_FN = partial(collate_fn, lfr=self.LFR)

        NUM_WORKERS = 4
        self.DATALOADER = DataLoader(self.dataset, num_workers=NUM_WORKERS, batch_size=self.BATCH_SIZE, pin_memory=True,
                                     collate_fn=self.COLLATE_FN)
        self.denormalize = NormalizeInverse(self.DATASET.mean, self.DATASET.std)

    def after(self):
        self.PATCH_SIZE = None
        self.LFR = None
        self.DATASET = None
        self.COLLATE_FN = None
        self.DATALOADER = None

    def test_get_collate_fn(self):

        batch = next(iter(self.DATASET))

        result = self.COLLATE_FN([batch])

        assert isinstance(result, Tuple), f"expected: Tuple. returned: {type(result)}"
        assert len(result) == 2, f"expected: 3, returned: {len(result)}"
        assert isinstance(result[0][0], float), f"expected: float, returned: {type(result[0])}"
        assert isinstance(result[0][1], torch.Tensor), f"expected: float, returned: {type(result[0])}"
        assert isinstance(result[0][2], torch.Tensor), f"expected: float, returned: {type(result[0])}"

    def test_dataset(self):
        from tqdm.auto import tqdm
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid

        for index, ((scale, low_res_batch_i_minus_1, low_res_batch_i), pyramid_i) in enumerate(
                tqdm(self.DATALOADER)):
            if index == 0:
                low_res_batch_i = self.denormalize(low_res_batch_i)
                pyramid_i = self.denormalize(pyramid_i)

                MAX_IMAGES = 16
                N_ROW = 4

                plt.figure(figsize=(10, 30))

                plt.subplot(211)
                plt.imshow(make_grid(low_res_batch_i[:MAX_IMAGES], nrow=N_ROW).permute(1, 2, 0))
                plt.title("Low res i")

                plt.subplot(212)
                plt.imshow(make_grid(pyramid_i[:MAX_IMAGES], nrow=N_ROW).permute(1, 2, 0))
                plt.title("Ground truth i")

                plt.show()
            else:
                break


if __name__ == "__main__":
    unittest.main()
