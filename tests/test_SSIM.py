import unittest
import random

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from base.trainer.Trainer import Trainer
from base.metrics.SSIM import gaussian_filter, SSIM
from base.transforms.NormalizeInverse import NormalizeInverse
from tests.pytorch_test import PyTorchTest
from tests.util_for_testing import RandomDataset, NetworkOneInput


class TestSSIM(PyTorchTest):
    def test_gaussian_filter(self):
        target = torch.as_tensor([[[0.0947, 0.1183, 0.0947],
                                   [0.1183, 0.1478, 0.1183],
                                   [0.0947, 0.1183, 0.0947]]])
        kernel = gaussian_filter(3, 1.5)
        torch.isclose(kernel, target)

    def before(self):
        device = torch.device("cpu")
        input_size = (3, 24, 24)
        dataset = RandomDataset(1, input_size, input_size)
        self.dataloader = DataLoader(dataset, batch_size=32)
        loss = MSELoss().to(device)
        model = NetworkOneInput(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        self.trainer = Trainer("test", model, adam, loss, metric=SSIM(reduction='sum'),
                               device=device)

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
        ssim = SSIM(dynamic_range=1., reduction='mean')(i1, i1)
        self.assertAlmostEqual(ssim, 1.)

    def test_raise_error(self):
        i1 = torch.rand(3, 5, 5)
        i2 = torch.rand(3, 5, 12)
        ssim = SSIM(dynamic_range=1., reduction='mean')
        self.assertRaises(RuntimeError, ssim, i1, i2)

    def test_different(self):
        i1 = torch.randint(0, 255, (3, 48, 48)) * 1.0
        i2 = 255. - i1
        ssim = SSIM(dynamic_range=255, reduction='mean')
        self.assertAlmostEqual(ssim(i1, i2), -1, delta=0.05)

    def test_normalized(self):
        mean = torch.rand(3, 1, 1)
        std = torch.rand(3, 1, 1)
        i1 = torch.rand(3, 48, 48)
        i1 = ((i1 - mean) / std)
        i2 = i1
        ssim = SSIM(dynamic_range=1, reduction='mean')
        print("Standardized:", ssim(i1, i2))

        i1 = (i1 * std + mean).clip(0, 1)
        i2 = i1
        ssim = SSIM(dynamic_range=1, reduction='mean')
        print("Unstandardized:", ssim(i1, i2))

    def test_same_image(self):
        import PIL.Image as PIM
        from torchvision import transforms
        mean_std_dict = {'mean': [0.4484562277793884, 0.4374960660934448, 0.40452802181243896],
                         'std': [0.2436375468969345, 0.23301854729652405, 0.24241816997528076]}
        import matplotlib.pyplot as plt

        ssim = SSIM(dynamic_range=1, reduction='mean')

        i = str(random.randint(1, 300))
        zero_to_add = 4 - len(i)
        i = "".join(['0'] * zero_to_add) + i
        img_path = fR"D:\University\PROJECTS\MLCV\datasets\DIV2K\DIV2K_train_HR\{i}.png"

        to_tensor = transforms.ToTensor()
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**mean_std_dict)])
        denormalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**mean_std_dict),
                                          NormalizeInverse(max_pixel_value=1, **mean_std_dict)])

        with PIM.open(img_path) as img:
            xy = random.randint(10, 100)
            width = xy + 100
            height = xy + 100
            box = (xy, xy, width, height)
            img_read = img.crop(box)


        read_img_1 = to_tensor(img_read)
        value = ssim(read_img_1, read_img_1)
        self.assertAlmostEqual(value, 1.)

        fig, axes = plt.subplots(ncols=3)
        axes[0].imshow(read_img_1.permute(1, 2, 0))

        read_img_1 = normalize(img_read)
        value = ssim(read_img_1, read_img_1)
        self.assertAlmostEqual(value, 1.)
        axes[1].imshow(read_img_1.permute(1, 2, 0))

        read_img_1 = denormalize(img_read)
        value = ssim(read_img_1, read_img_1)
        self.assertAlmostEqual(value, 1.)
        axes[2].imshow(read_img_1.permute(1, 2, 0))
        plt.show()


if __name__ == "__main__":
    unittest.main()
