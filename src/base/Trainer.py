from __future__ import annotations

import operator
import re
from collections import defaultdict, namedtuple
from itertools import starmap
from typing import Any, NamedTuple

import torch
import torch.optim
import torch.utils.data
import torch.utils.tensorboard

import utility
from base import BaseModel
from base.Callback import Callback, ListCallback

from base.hints import Union, List, Tuple, Criterion, Metric, Dict, Callable

from tqdm.auto import tqdm

import sys
import logging

logging.basicConfig()
logger = logging.getLogger("Trainer")
logger.setLevel(logging.DEBUG)


# TODO: Review comments and docstring
# TODO: Callback tests
# TODO: Checkpoint experiment
# TODO: Implement resuming


# endregion

class Trainer:
    """ Class for training a BaseModel network

    Args:
        experiment: name of the experiment (define the name of the log folder)
        model: model to train
        optimizer: optimizer to use
        lr_scheduler: learning rate to use
        callback: callbacks to call
    """

    def __init__(self,
                 experiment: str,
                 model: BaseModel,
                 optimizer: torch.optim.Optimizer,
                 criterion: Criterion,
                 metric: Union[Metric, List[Metric], Dict[str, Metric]] = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 callback: Union[Callback, ListCallback] = None,
                 device: torch.device = None):

        # set the experiment name
        self.experiment = experiment

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # if device is None model and criterion are not in the same device
            self.model = model.to(device)
            self.criterion = criterion.to(device)
            # move metric if is a module
            if isinstance(metric, torch.nn.Module):
                self.metric = metric.to(device)
            elif (isinstance(metric, List)):
                self.metric = [metric_fn.to(device) if isinstance(metric_fn, torch.nn.Module) else metric_fn for
                               metric_fn in metric]
            elif (isinstance(metric, Dict)):
                self.metric = {key: (metric_fn.to(device) if isinstance(metric_fn, torch.nn.Module) else metric_fn) for
                               key, metric_fn in metric.items()}
        else:
            self.model = model
            self.criterion = criterion
            self.metric = metric

        # set the History and TrainingState instance attribute
        # TrainingState(loss,metric1,...,metric_n)
        self.TrainingState = self._create_training_state_obj(metric)

        # set the device for moving batches
        self.device = device

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.callback = callback

        # flag used for stopping the training
        self.stop = False

        # TODO Move to TensorbordCallback
        # define where log and files events will be saved
        self.log_dir = self.model.model_dir / experiment
        self.tensorboard = torch.utils.tensorboard.SummaryWriter(str(self.log_dir))

        # define the logger
        # log = self.model.model_dir / "logs" / f"{experiment}.log"
        # self.logger = logging.getLogger(f"{__name__}_{model.name}")
        # self.logger.setLevel(logging.INFO)
        # consoleHandler = logging.StreamHandler(sys.stdout)
        # fileHandler = logging.FileHandler(filename=log, mode="w")
        # self.logger.addHandler(consoleHandler)
        # self.logger.addHandler(fileHandler)

        # self.logger.info(f"Initialized Trainer for {experiment} with {model.name}")
        logger.debug(f"Init trainer class for {experiment} with {model.name}")

    def stop(self, sender=None):
        logger.debug(f"Trainer stopped by {'itself' if sender is None else sender.__class__.__name__}")
        self.stop = True

    def train(self, loader: torch.utils.data.DataLoader) -> 'TrainingState':
        """Define the train loop for one epoch"""

        logger.debug("Set model in training mode")
        self.model.train()

        epoch_loss = 0
        total_samples = 0

        # define the epoch_metric type
        if self.metric:
            epoch_metric = Trainer._init_metric_obj(self.metric)

        for index_batch, (X, y) in enumerate(tqdm(loader, total=len(loader), unit="batch", leave=False)):
            logger.debug(f"Input received: {type(X)}")
            logger.debug(f"Target received: {type(y)}")

            # erase all saved gradients
            logger.debug("Erasing gradients inside the optimizer")
            self.optimizer.zero_grad()

            # extract batch size from X
            logger.debug("Extracting batch size")
            BATCH_SIZE = self._get_batch_size(X)
            total_samples += BATCH_SIZE

            # move batch and targets to device
            logger.debug("Move X and y to the selected device")
            X, y = self._move_to_device(X, y)

            # forward the input in the correct way:
            # e.g:
            # forward(input)
            # forward(input1,input2,input3) we have to unfold the list or the tuple
            logger.debug("Forwarding the input")
            out = self._forward(X)
            self._assert_out_requires_grad(out)

            # TODO: Handle multiple output
            # compute the loss
            logger.debug("Computing the loss")
            if not out.size() == y.size():
                raise RuntimeError(f"Size mismatching: (out) received {out.size()}, (y) expected: {y.size()}")
            loss = self.criterion(out, y)

            # compute the gradients
            logger.debug("Computing the gradients")
            loss.backward()

            # update the parameters
            logger.debug("Updating the parameters")
            self.optimizer.step()

            # update the loss
            # loss.item() contains the loss averaged in the batch
            # * X.size(0) we get the non averaged loss in the batch
            logger.debug("Updating the epoch loss counter")
            epoch_loss += loss.item() * BATCH_SIZE

            # compute the metric
            logger.debug("Computing metric")
            if self.metric:
                computed_metric = Trainer._compute_metric(self.metric, prediction=out, target=y)
                # add metric computed previously with the one computed now
                epoch_metric = Trainer._execute_operation(operator.add, epoch_metric, computed_metric)

        # average the metric
        if self.metric:
            epoch_metric = Trainer._execute_operation(operator.__truediv__, epoch_metric, len(loader))

        # return the averaged loss in the whole dataset and metrics
        logger.debug("Returning loss and metrics of the whole dataset")
        return self._return_training_state(epoch_loss=epoch_loss, epoch_metric=epoch_metric)

    def validation(self, loader: torch.utils.data.DataLoader) -> 'TrainingState':
        self.model.eval()

        epoch_loss = 0
        total_samples = 0

        if self.metric:
            epoch_metric = Trainer._init_metric_obj(self.metric)

        for index_batch, (X, y) in enumerate(tqdm(loader, total=len(loader), unit="batch", leave=False)):
            BATCH_SIZE = self._get_batch_size(X)
            total_samples += BATCH_SIZE

            X, y = self._move_to_device(X, y)

            out = self._forward(X)
            self._assert_out_requires_grad(out)

            assert out.size() == y.size()
            loss = self.criterion(out, y)

            epoch_loss += loss.item() * BATCH_SIZE
            if self.metric:
                computed_metric = Trainer._compute_metric(self.metric, prediction=out, target=y)
                epoch_metric = Trainer._execute_operation(operator.add, epoch_metric, computed_metric)

        epoch_loss = epoch_loss / total_samples

        if self.metric:
            epoch_metric = Trainer._execute_operation(operator.__truediv__, epoch_metric, len(loader))

        return self._return_training_state(epoch_loss=epoch_loss, epoch_metric=epoch_metric)

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, epochs: int):

        # TODO Implement resuming
        for epoch in range(1, epochs + 1):

            if self.stop:
                logger.info("Stopping training.")
                break

            if self.callback:
                self.callback.start_epoch(self._create_args_for_callback(epoch=epoch))

            train_state = self.train(train_loader)
            val_state = self.validation(val_loader)

            print(Trainer._return_loss_and_metric_formatted(train_state, train=True))
            print(Trainer._return_loss_and_metric_formatted(val_state, train=False))

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if self.callback:
                self.callback.end_epoch(
                    self._create_args_for_callback(epoch=epoch, train_state=train_state, val_state=val_state))

        # TODO Implement history
        return None

    @staticmethod
    def _create_training_state_obj(metric):
        field_names = 'loss'
        if metric:
            names = Trainer._get_metric_name(metric)
            if isinstance(names, str):
                names = Trainer._validate_field_names(names)
            elif isinstance(names, List):
                names = [Trainer._validate_field_names(name) for name in names]
            else:
                raise RuntimeError("Invalid field names type")

            field_names = ['loss'] + ([names] if not isinstance(names, List) else names)
        return namedtuple("TrainingState", field_names)

    def _return_training_state(self, epoch_loss, epoch_metric):
        if self.metric:
            if isinstance(epoch_metric, Dict):
                epoch_metric = epoch_metric.values()

            if isinstance(epoch_metric, float):
                return self.TrainingState(epoch_loss, epoch_metric)
            else:
                return self.TrainingState(epoch_loss, *epoch_metric)
        else:
            return self.TrainingState(epoch_loss)

    @staticmethod
    def _validate_field_names(name):
        return re.sub("[^\w\d]+", '_', name.rstrip())

    @staticmethod
    def _return_loss_and_metric_formatted(state: 'TrainingState', train: bool):
        split = "train" if train else "val"
        return " ".join(
            f"{split}_{key}: {value}" for key, value in state._asdict().items()
        )

    @staticmethod
    def _get_metric_name(metric):
        if isinstance(metric, torch.nn.Module):
            return str(metric)
        elif isinstance(metric, Callable):
            return str(metric.__name__)
        elif isinstance(metric, List):
            return [Trainer._get_metric_name(metric_fn) for metric_fn in metric]
        elif isinstance(metric, Dict):
            return [key for key in metric]

    def _forward(self, X):
        if isinstance(X, torch.Tensor):
            return self.model.train_step(X) if self.model.training else self.model.val_step(X)
        elif isinstance(X, (List, Tuple)):
            return self.model.train_step(*X) if self.model.training else self.model.val_step(*X)
        else:
            raise RuntimeError(f"Invalid type: {type(X)}")

    @staticmethod
    def _init_metric_obj(metric):
        # if isinstance(self.metric, Metric):
        # It's a python design decision to not use GenericAlias
        if isinstance(metric, (Callable, torch.nn.Module)):
            logger.debug("metric is a function")
            return 0
        elif isinstance(metric, List):
            logger.debug("metric is a list")
            return [0 for _ in range(len(metric))]
        elif isinstance(metric, Dict):
            logger.debug("metric is a dict")
            return {key: 0 for key in metric}
        else:
            raise RuntimeError(
                f"Invalid type: received {type(metric)}, expected: Metric,List[Metric],Dict[str,Metric]")

    @staticmethod
    def _compute_metric(metric, prediction: torch.Tensor, target: torch.Tensor):
        if isinstance(prediction, torch.Tensor) and isinstance(target, torch.Tensor):
            if isinstance(metric, (Callable, torch.nn.Module)):
                return metric(prediction, target)
            elif isinstance(metric, List):
                return [m(prediction, target) for m in metric]
            elif isinstance(metric, Dict):
                return {key: metric_fn(prediction, target) for key, metric_fn in metric.items()}
            else:
                raise RuntimeError(f"Unsupported type: {type(metric)}")
        else:
            # TODO: Improve this
            raise RuntimeError(
                f"Unsupported type: (prediction){type(prediction)}, (target){type(target)}, expected Tensor")

    def _assert_out_requires_grad(self, out):
        if isinstance(out, torch.Tensor):
            assert out.requires_grad == self.model.training
        elif isinstance(out, (List, Tuple)):
            for elem in out:
                if isinstance(elem, torch.Tensor):
                    assert elem.requires_grad == self.model.training

    def _get_batch_size(self, X):
        if isinstance(X, torch.Tensor):
            return X.size(0)
        elif isinstance(X, (List, Tuple)):
            sizes = [elem.size(0) for elem in X if isinstance(elem, torch.Tensor)]
            return sizes[0]
        else:
            raise RuntimeError(f"Invalid type: {type(X)}")

    def _move_to_device(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.to(self.device)
        elif isinstance(X, Tuple):
            X = tuple([elem.to(self.device) if isinstance(elem, torch.Tensor) else elem for elem in X])
        elif isinstance(X, List):
            X = [elem.to(self.device) if isinstance(elem, torch.Tensor) else elem for elem in X]
        else:
            raise RuntimeError(f"Invalid type: {type(X)}")

        if isinstance(y, torch.Tensor):
            y = y.to(self.device)
        elif isinstance(X, Tuple):
            y = tuple([elem.to(self.device) if isinstance(elem, torch.Tensor) else elem for elem in y])
        elif isinstance(X, List):
            y = [elem.to(self.device) if isinstance(elem, torch.Tensor) else elem for elem in y]
        else:
            raise RuntimeError(f"Invalid type: {type(y)}")

        return X, y

    def _create_args_for_callback(self, **kwargs):
        """ Return a dictionary of arguments to pass to callbacks

        Args:
            **kwargs: named arguments to pass

        Returns:
            {
                "experiment" : self.experiment,
                "model" : self.model,
                "optimizer" self.optimizer,
                **kwargs
        """

        base_kwargs = {
            "experiment": self.experiment,
            "model": self.model,
            "optimizer": self.optimizer,
        }

        return {**base_kwargs, **kwargs}

    @staticmethod
    def _execute_operation(operation, first, second):
        """ Execute operation between first and second (handling operation element wise if they are List or Dict)
        Args:
            operation: scalar operation to do (e.g. add)
            first
            second
        """

        def __for_list(current: float, computed: float):
            return operation(current, computed)

        def __for_dict(current: Tuple[str, float], computed: Tuple[str, float]):
            key_current, value_current = current
            key_computed, value_computed = computed
            assert key_current == key_computed
            return (key_current, operation(value_current, value_computed))

        # if current is float apply simply the operation
        if isinstance(first, (int, float)):
            return operation(first, second)
        # if current is a list apply operation element wise
        elif isinstance(first, List):
            # in case other is a int or float, create a list with other repeated as much as len(current)
            if isinstance(second, (int, float)):
                second = [second] * len(first)
            return list(starmap(__for_list, zip(first, second)))
        elif isinstance(first, Dict):
            # create a copy of current dict using other for all keys in order to do element wise operation
            if isinstance(second, (int, float)):
                second = {key: second for key in first}
            return dict(starmap(__for_dict, zip(first.items(), second.items())))
        else:
            raise RuntimeError(type(first), type(second))

    def save_model(self):
        try:
            dict_to_save = {"param": self.model.__dict__,
                            "state": self.model.state_dict()
                            }
            model_path = self.model.model_dir / f"{self.model.name}_state_dict.pytorch"
            torch.save(dict_to_save, model_path)
        except:
            logger.error("An error occurred saving the model.")

    def save_experiment(self, last_epoch: int):
        try:
            checkpoint_path = self.log_dir / f"trainer_{last_epoch}.pytorch"
            logger.debug(f"Save trainer information at: {checkpoint_path}")
            what_to_save = {
                "experiment": self.experiment,
                "model": self.model,
                "criterion": self.criterion,
                "optimizer": self.optimizer,
                "lr_scheduler": self.lr_scheduler,
                "metric": self.metric,
                "callback": self.callback,
                "device": self.device
            }

            torch.save(what_to_save, checkpoint_path)
        except AttributeError as ae:
            if "Can't pickle local object":
                new_message = str(ae) + "\n"
                new_message += "Insert the unpickable function inside a separate module and import the function."
                logger.error(new_message)
                ae = RuntimeError(new_message)
            raise ae
        logger.debug("Information saved.")

    @classmethod
    def load_experiment(cls, base_model_dir: str, experiment: str, last_epoch: int):
        checkpoint_path = utility.get_models_dir() / base_model_dir / experiment / f"trainer_{last_epoch}.pytorch"
        if not checkpoint_path.exists():
            raise RuntimeError(f"{checkpoint_path} doesn't exists.")

        what_to_load = torch.load(checkpoint_path)
        logger.debug(what_to_load)
        return cls(experiment=what_to_load['experiment'],
                   model=what_to_load['model'],
                   optimizer=what_to_load['optimizer'],
                   criterion=what_to_load['criterion'],
                   metric=what_to_load['metric'],
                   lr_scheduler=what_to_load['lr_scheduler'],
                   callback=what_to_load['callback'],
                   device=what_to_load['device'])
