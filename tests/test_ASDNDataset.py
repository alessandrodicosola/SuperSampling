from functools import partial
from typing import Tuple
from unittest import TestCase
import unittest
from torch.utils.data import DataLoader
import torch
from datasets.ASDNDataset import ASDNDataset, collate_fn, NormalizeInverse
from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation


class TestASDNDataset(TestCase):
    def test_get_collate_fn(self):
        PATCH_SIZE = 48
        LFR = LaplacianFrequencyRepresentation(1, 2, 11)
        DATASET = ASDNDataset("DIV2K_valid_HR", patch_size=PATCH_SIZE, lfr=LFR)

        COLLATE_FN = partial(collate_fn, lfr=LFR)

        batch = next(iter(DATASET))

        result = COLLATE_FN([batch])

        assert isinstance(result, Tuple), f"expected: Tuple. returned: {type(result)}"
        assert len(result) == 3, f"expected: 2, returned: {len(result)}"
        assert len(result[0]) == 2, f"expected: 2, returned: {len(result[0])}"
        assert len(result[1]) == 2, f"expected: 2, returned: {len(result[1])}"
        assert len(result[2]) == 2, f"expected: 2, returned: {len(result[2])}"

    def test_dataset(self):
        from tqdm.auto import tqdm
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid

        START, END, COUNT = 1, 2, 11

        PATCH_SIZE = 48

        BATCH_SIZE = 8

        NUM_WORKERS = 4

        LFR = LaplacianFrequencyRepresentation(START, END, COUNT)

        dataset = ASDNDataset("DIV2K_valid_HR", patch_size=PATCH_SIZE, lfr=LFR)

        COLLATE_FN = partial(collate_fn, lfr=LFR)

        dataloader = DataLoader(dataset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, pin_memory=True,
                                collate_fn=COLLATE_FN)

        denormalize = NormalizeInverse(dataset.mean, dataset.std)

        for index, ((scale), (low_res_batch_i_minus_1, pyramid_i_minus_1), (low_res_batch_i, pyramid_i)) in enumerate(
                tqdm(dataloader)):
            if index == 0:
                low_res_batch_i_minus_1 = denormalize(low_res_batch_i_minus_1)
                pyramid_i_minus_1 = denormalize(pyramid_i_minus_1)

                low_res_batch_i = denormalize(low_res_batch_i)
                pyramid_i = denormalize(pyramid_i)

                MAX_IMAGES = 16
                N_ROW = 4

                plt.figure(figsize=(10, 30))

                plt.subplot(411)
                plt.imshow(make_grid(low_res_batch_i_minus_1[:MAX_IMAGES], nrow=N_ROW).permute(1, 2, 0))
                plt.title("Low res i-1")

                plt.subplot(412)
                plt.imshow(make_grid(pyramid_i_minus_1[:MAX_IMAGES], nrow=N_ROW).permute(1, 2, 0))
                plt.title("Ground truth i-1")

                plt.subplot(413)
                plt.imshow(make_grid(low_res_batch_i[:MAX_IMAGES], nrow=N_ROW).permute(1, 2, 0))
                plt.title("Low res i")

                plt.subplot(414)
                plt.imshow(make_grid(pyramid_i[:MAX_IMAGES], nrow=N_ROW).permute(1, 2, 0))
                plt.title("Ground truth i")

                plt.show()
            else:
                break


if __name__ == "__main__":
    unittest.main()
