from unittest import TestCase

from datasets import TestDataset
from datasets.ASDNDataset import ASDNDataset

from utility import get_datasets_dir

import matplotlib.pyplot as plt


class TestTestDataset(TestCase):

    def test_getitem(self):
        dataset = TestDataset.TestDataset([2, 3.5, 4], get_datasets_dir() / "Test" / "Urban100")

        scale, lr, gt = next(iter(dataset))
        denorm = ASDNDataset.denormalize_fn()

        print(scale)

        fig, axes = plt.subplots(ncols=2)
        axes[0].imshow(denorm(lr).permute(1, 2, 0))
        axes[1].imshow(gt.permute(1, 2, 0))
        fig.suptitle(f"Scale used: {scale}")
        plt.show()
