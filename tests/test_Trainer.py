import operator
import unittest
from functools import partial
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from base.callbacks.EarlyStoppingCallback import EarlyStoppingCallback
from base.trainer.Trainer import Trainer

from datasets.ASDNDataset import ASDNDataset, create_batch_for_training
from models.ASDN import ASDN
from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation
from tests.pytorch_test import PyTorchTest
from tests.util_for_testing import RandomDataset, NetworkOneInput, NetworkTwoInputs

class TestForwardASDN(PyTorchTest):
    def before(self):
        LFR = LaplacianFrequencyRepresentation(1, 2, 11)
        COLLATE_FN = partial(create_batch_for_training, lfr=self.LFR)

        VAL_DATASET = ASDNDataset("DIV2K_valid_HR", 24, LFR)
        self.VAL_DATALOADER = DataLoader(VAL_DATASET, 64, False, num_workers=4, collate_fn=COLLATE_FN)

        TRAIN_DATASET = ASDNDataset("DIV2K_train_HR", 24, LFR)
        self.TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, 32, True, num_workers=4, collate_fn=COLLATE_FN)

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ASDN_ = ASDN(3, LFR, n_dab=3, n_intra_layers=3, out_compressed_channels=32, out_channels_dab=8,
                     intra_layer_output_features=8).to(self.DEVICE)
        ADAM = Adam(ASDN_.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        self.TRAINER = Trainer("test", ASDN_, ADAM, MSELoss().to(DEVICE), device=DEVICE)

    def after(self):
        self.VAL_DATALOADER = None
        self.TRAIN_DATALOADER = None
        self.TRAINER = None

    @unittest.skip("EXPENSIVE")
    def test_forward(self):
        self.TRAINER.fit(self.TRAIN_DATALOADER, self.VAL_DATALOADER, 1)


class TestForwardBaseModuleOneInput(PyTorchTest):
    def before(self):
        device = torch.device('cpu')
        input_size = (3, 24, 24)
        dataset = RandomDataset(1, input_size, input_size)
        self.dataloader = DataLoader(dataset, batch_size=32)
        loss = MSELoss().to(device)
        model = NetworkOneInput(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        self.trainer = Trainer("test", model, adam, loss, device=device)

    def after(self):
        self.dataloader = None
        self.trainer = None

    def test_forward(self):
        self.trainer.fit(self.dataloader, self.dataloader, 5)


class TestForwardBaseModuleTwoInputs(PyTorchTest):

    def collate_fn(self, batch):
        inputs, target = zip(*batch)
        input1, input2 = zip(*inputs)
        return (torch.stack(input1, dim=0), torch.stack(input2, dim=0)), torch.stack(target, dim=0)

    def before(self):
        device = torch.device('cpu')
        input_size = (3, 24, 24)
        dataset = RandomDataset(2, input_size, input_size)
        self.dataloader = DataLoader(dataset, batch_size=32, collate_fn=self.collate_fn)
        loss = MSELoss().to(device)
        model = NetworkTwoInputs(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        self.trainer = Trainer("test", model, adam, loss, device=device)

    def after(self):
        self.dataloader = None
        self.trainer = None

    def test_forward(self):
        self.trainer.fit(self.dataloader, self.dataloader, 5)


class TestMetric(PyTorchTest):
    def before(self):
        pass

    def after(self):
        pass

    def metric1(t1: torch.Tensor, t2: torch.Tensor):
        return ((t1 - t2) ** 2).sum().item()

    def test_single_metric(self):
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


class Test__execute_operation(PyTorchTest):
    def before(self):
        pass

    def after(self):
        pass

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


class Test_save_load_Trainer(PyTorchTest):

    def before(self):
        metric_dict = {"my awesome metric1": metric1, "the less interesting metric2": metric2}
        device = torch.device('cpu')
        input_size = (3, 24, 24)
        dataset = RandomDataset(1, input_size, input_size)
        self.dataloader = DataLoader(dataset, batch_size=32)
        loss = MSELoss().to(device)
        model = NetworkOneInput(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(adam, 3)
        self.trainer = Trainer("test", model, adam, loss, metric=metric_dict, lr_scheduler=lr_scheduler, device=device)

    def after(self):
        self.dataloader = None
        self.trainer = None
        del self.trainer

    def test_save_experiment_raise(self):
        def metric1(t1: torch.Tensor, t2: torch.Tensor):
            return ((t1 - t2) ** 2).sum().item()

        def metric2(t1: torch.Tensor, t2: torch.Tensor):
            return ((t1 - t2) ** 2).sum().item()

        loss = MSELoss()
        input_size = (3, 24, 24)
        model = NetworkOneInput(input_size, 32)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        trainer = Trainer("test", model, adam, loss, metric={"metri1": metric1, "metric2": metric2})
        self.assertRaises(RuntimeError, trainer.save_experiment, 1, 1e-5, None)

    def test_save_experiment_not_raise(self):
        try:
            self.trainer.save_experiment(1, 1e-4, None)
        except:
            self.fail("Error occurred")

    def test_load_experiment(self):
        history = self.trainer.fit(self.dataloader, self.dataloader, 10)
        self.trainer.save_experiment(1, 23e-5, history)

        trainer = self.trainer
        test_trainer = Trainer.load_experiment("NetworkOneInput", "test", 1)

        for p1, p2 in zip(trainer.model.parameters(), test_trainer.model.parameters()):
            self.assertTrue(self.Tensors_are_equal(p1, p2))

        self.state_dicts_are_equal(trainer.model.state_dict(), test_trainer.model.state_dict())
        self.state_dicts_are_equal(trainer.optimizer.state_dict(), test_trainer.optimizer.state_dict())
        self.state_dicts_are_equal(trainer.lr_scheduler.state_dict(), test_trainer.lr_scheduler.state_dict())

        self.cls_are_equal(trainer.criterion, test_trainer.criterion)
        self.cls_are_equal(trainer.metric, test_trainer.metric)
        self.cls_are_equal(trainer.callback, test_trainer.callback)

        self.assertEqual(trainer.device, test_trainer.device)
        self.assertEqual(trainer.model_info, test_trainer.model_info)
        self.assertEqual(trainer.log_dir, test_trainer.log_dir)

    def test_history(self):
        epochs = 5
        history = self.trainer.fit(self.dataloader, self.dataloader, epochs)
        self.assertTrue(len(history.train) == epochs)
        self.assertTrue(len(history.val) == epochs)


class TestTrainer_Checkpoint(PyTorchTest):
    def before(self):
        metric_dict = {"my awesome metric1": metric1, "the less interesting metric2": metric2}
        self.epochs = 25

        device = torch.device('cpu')
        input_size = (3, 24, 24)
        dataset = RandomDataset(1, input_size, input_size, size_dataset=128)
        self.dataloader = DataLoader(dataset, batch_size=32)
        loss = MSELoss().to(device)
        model = NetworkOneInput(input_size, 32).to(device)
        adam = Adam(model.parameters(), 1e-3, betas=(0.99, 0.999), eps=1e-8)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(adam, 10)
        self.trainer = Trainer("test", model, adam, loss, metric=metric_dict, lr_scheduler=lr_scheduler, device=device,
                               callback=EarlyStoppingCallback(patience=10))

    def after(self):
        self.dataloader = None
        self.trainer = None

    def test_EarlyStopping(self):
        history = self.trainer.fit(self.dataloader, self.dataloader, self.epochs)
        # after running once the early stopping happens at 25
        self.assertTrue(len(history.train) == 25)
        self.assertTrue(len(history.val) == 25)


if __name__ == "__main__":
    unittest.main()
