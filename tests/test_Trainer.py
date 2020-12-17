import operator
import unittest
from functools import partial
from unittest import TestCase
import torch
from torch.nn import MSELoss, Module
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset

from base import BaseModel
from base.Trainer import Trainer
from base.hints import Criterion

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


class TestForward(TestCase):

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


class TestMetric(unittest.TestCase):
    def test_single_metric(self):
        def metric1(t1: torch.Tensor, t2: torch.Tensor):
            return ((t1 - t2) ** 2).sum().item()

        device = torch.device('cpu')
        input_size = (3, 24, 24)
        dataset = RandomDataset(1, input_size, input_size)
        dataloader = DataLoader(dataset, batch_size=32)
        loss = MSELoss().to(device)
        model = NetworkOneInput(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        trainer = Trainer("test", model, adam, loss, metric=metric1, device=device)
        trainer.fit(dataloader, dataloader, 5)

    def test_list_metric(self):
        def metric1(t1: torch.Tensor, t2: torch.Tensor):
            return ((t1 - t2) ** 2).sum().item()

        def metric2(t1: torch.Tensor, t2: torch.Tensor):
            return ((t1 - t2) ** 2).sum().item()

        device = torch.device('cpu')
        input_size = (3, 24, 24)
        dataset = RandomDataset(1, input_size, input_size)
        dataloader = DataLoader(dataset, batch_size=32)
        loss = MSELoss().to(device)
        model = NetworkOneInput(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        trainer = Trainer("test", model, adam, loss, metric=[metric1, metric2], device=device)
        trainer.fit(dataloader, dataloader, 5)

    def test_dict_metric(self):
        def metric1(t1: torch.Tensor, t2: torch.Tensor):
            return ((t1 - t2) ** 2).sum().item()

        def metric2(t1: torch.Tensor, t2: torch.Tensor):
            return ((t1 - t2) ** 2).sum().item()

        metric_dict = {"my awesome metric1": metric1, "the less interesting metric2": metric2}
        device = torch.device('cpu')
        input_size = (3, 24, 24)
        dataset = RandomDataset(1, input_size, input_size)
        dataloader = DataLoader(dataset, batch_size=32)
        loss = MSELoss().to(device)
        model = NetworkOneInput(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        trainer = Trainer("test", model, adam, loss, metric=metric_dict, device=device)
        trainer.fit(dataloader, dataloader, 5)


class Test__execute_operation(TestCase):
    def test__execute_operation_int(self):
        self.assertTrue(Trainer._execute_operation(operator.add, 1, 2), 3)
        self.assertAlmostEqual(Trainer._execute_operation(operator.__truediv__, 1, 2), 0.5)

    def test__execute_operation_float(self):
        self.assertTrue(Trainer._execute_operation(operator.add, 1.0, 2.0), 3.0)
        self.assertAlmostEqual(Trainer._execute_operation(operator.__truediv__, 1.0, 2.0), 0.5)

    def test__execute_operation_float_int(self):
        self.assertTrue(Trainer._execute_operation(operator.add, 1.0, 2), 3)
        self.assertAlmostEqual(Trainer._execute_operation(operator.__truediv__, 1.0, 2), 0.5)

    def test__execute_operation_list(self):
        self.assertTrue(Trainer._execute_operation(operator.add, [1, 2, 3], [1, 2, 3]), [2, 4, 6])

        true = [1, 1, 1]
        out = Trainer._execute_operation(operator.__truediv__, [1, 2, 3], [1, 2, 3])
        for elem, t in zip(out, true):
            self.assertAlmostEqual(elem, t)

        true = [1, 2, 3]
        out = Trainer._execute_operation(operator.__truediv__, [1, 2, 3], 1)
        for elem, t in zip(out, true):
            self.assertAlmostEqual(elem, t)

        true = [1.0, 2.0, 3.0]
        out = Trainer._execute_operation(operator.__truediv__, [1, 2, 3], 1.0)
        for elem, t in zip(out, true):
            self.assertAlmostEqual(elem, t)

        true = [i / 2 for i in range(1, 3 + 1)]
        out = Trainer._execute_operation(operator.__truediv__, [1, 2, 3], 2)
        for elem, t in zip(out, true):
            self.assertAlmostEqual(elem, t)

    def test__execute_operation_dict(self):
        d1 = {"metric1": 0.04, "metric2": 0.6}
        d2 = {"metric1": 0.04, "metric2": 1.2}
        out = Trainer._execute_operation(operator.add, d1, d2)
        self.assertTrue(out, {"metric1": 0.08, "metric2": 1.8})


def metric1(t1: torch.Tensor, t2: torch.Tensor):
    return ((t1 - t2) ** 2).sum().item()


def metric2(t1: torch.Tensor, t2: torch.Tensor):
    return ((t1 - t2) ** 2).sum().item()


class Test_save_load_Trainer(TestCase):

    def test_save_experiment_raise(self):
        def metric1(t1: torch.Tensor, t2: torch.Tensor):
            return ((t1 - t2) ** 2).sum().item()

        def metric2(t1: torch.Tensor, t2: torch.Tensor):
            return ((t1 - t2) ** 2).sum().item()

        metric_dict = {"my awesome metric1": metric1, "the less interesting metric2": metric2}
        device = torch.device('cpu')
        input_size = (3, 24, 24)
        dataset = RandomDataset(1, input_size, input_size)
        dataloader = DataLoader(dataset, batch_size=32)
        loss = MSELoss().to(device)
        model = NetworkOneInput(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        trainer = Trainer("test", model, adam, loss, metric=metric_dict, device=device)
        self.assertRaises(AttributeError, trainer.save_experiment, 1)

    def test_save_experiment_not_raise(self):
        metric_dict = {"my awesome metric1": metric1, "the less interesting metric2": metric2}
        device = torch.device('cpu')
        input_size = (3, 24, 24)
        dataset = RandomDataset(1, input_size, input_size)
        dataloader = DataLoader(dataset, batch_size=32)
        loss = MSELoss().to(device)
        model = NetworkOneInput(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        trainer = Trainer("test", model, adam, loss, metric=metric_dict, device=device)

        try:
            trainer.save_experiment(1)
        except:
            self.fail("Error occurred")

    # TODO Heavy testing
    def test_load_experiment(self):
        self.maxDiff = None
        torch.manual_seed(1)

        metric_dict = {"my awesome metric1": metric1, "the less interesting metric2": metric2}
        device = torch.device('cpu')
        input_size = (3, 24, 24)
        dataset = RandomDataset(1, input_size, input_size)
        dataloader = DataLoader(dataset, batch_size=32)
        loss = MSELoss().to(device)
        model = NetworkOneInput(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(adam, 10)
        trainer = Trainer("test", model, adam, loss, metric=metric_dict, lr_scheduler=lr_scheduler, device=device)

        try:
            trainer.save_experiment(1)
            history = trainer.fit(dataloader,dataloader,10)
        except:
            self.fail("Error occurred")

        test_trainer = Trainer.load_experiment("NetworkOneInput", "test", 1)

        for p1,p2 in zip(trainer.model.parameters(),test_trainer.model.parameters()):
            if torch.eq(p1,p2).sum() > 0:
                self.fail("p1 and p2 are not equal")





if __name__ == "__main__":
    unittest.main()
