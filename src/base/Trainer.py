from __future__ import annotations

import operator
import re

from collections import namedtuple
from itertools import starmap
from pathlib import Path

import torch
import torch.optim
import torch.utils.data
import torch.utils.tensorboard
import torch.optim.lr_scheduler

from typing import NoReturn, Union, List, Dict, NamedTuple, Callable, Tuple

import typing

import utility

from base import BaseModule
from base.Callbacks.Callback import Callback, ListCallback

from tqdm.auto import tqdm

import logging

from base.Callbacks import Callback, ListCallback, StdOutCallback
from base.Callbacks.TensorboardCallback import TensorboardCallback
from base.hints import Criterion, Metric

logging.basicConfig()
logger = logging.getLogger("Trainer")
logger.setLevel(logging.DEBUG)

# TODO: Callbacks


ModelInformation = typing.NamedTuple('ModelInformation', [
    ("name", str),
    ("model_dir", Path)
])


class Trainer:
    """Define a class for handling training

    Args:
        experiment: name of the experiment
        model: model to train
        optimizer: optimizer to use
        criterion: criterion to use
        metric: metric or list of metric or dict of metric to use. (Default: None)
        lr_scheduler: scheduler to use. (Default: None)
        callback: callback(s) to use. (Default: None)
        device: device where compute the operation. If none a device will be used automatically based on ones available. (Default: None)
    """

    def __init__(self,
                 experiment: str,
                 model: BaseModule,
                 optimizer: torch.optim.Optimizer,
                 criterion: Criterion,
                 metric: Union[Metric, List[Metric], Dict[str, Metric]] = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 callback: Union[Callback, List[Callback]] = None,
                 device: torch.device = None):

        # register model information
        self.model_info = Trainer._register_model_information(model)

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

        # set TrainingState type (it's dynamic: it changes with metric)
        # TrainingState(loss,metric1,...,metric_n)
        # set HistoryState (which is based on TrainingState)
        self.TrainingState = Trainer._register_training_state_type(metric)
        self.HistoryState = self._register_history_state_type()

        # set the device for moving batches
        self.device = device

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.callback = callback

        # flag used for stopping the training
        self.stop = False

        # define where log and files events will be saved
        self.log_dir = self.model_info.model_dir / experiment

        self.callback = self._register_callbacks(callback)

        logger.debug(f"Init trainer class for {self.experiment} with {self.model_info.name}")

    def _register_model_information(model: torch.nn.Module):
        """
        Save model information inside the Trainer

        Returns:
            ModelInformation namedtuple
        """
        name = model.__class__.__name__
        model_dir = utility.get_models_dir() / name
        return ModelInformation(name=name,
                                model_dir=model_dir)

    def _register_callbacks(self, optional_callback: Union[Callback, List[Callback]]):
        default_callbacks = [
            StdOutCallback(),
            TensorboardCallback(str(self.log_dir))
        ]

        if optional_callback:
            if isinstance(optional_callback, Callback):
                optional_callback = [optional_callback]

            default_callbacks = default_callbacks + optional_callback

        return ListCallback.from_list(default_callbacks)

    def _stop_fn(self, sender=None) -> NoReturn:
        """Function for stopping the training

        Args:
            sender: object which call the function
        """
        logger.debug(f"Trainer stopped by {'itself' if sender is None else sender.__class__.__name__}")
        self.stop = True

    def train(self, loader: torch.utils.data.DataLoader, **kwargs) -> 'TrainingState':
        """Define the train loop for one epoch

        Args:
            loader: DataLoader for extracting train batches

        Keyword Args:
            epoch: current epoch

        Returns:
            TrainingState (a namedtuple generated dynamically) which contains loss and metric(s) (if self.metric is not None)

        See Also
            :class:`~Trainer._register_training_state_type(metric)` for understanding the return type
        """

        logger.debug("Set model in training mode")
        self.model.train()

        epoch_loss = 0
        total_samples = 0

        # define the epoch_metric type
        epoch_metric = Trainer._init_metric_obj(self.metric) if self.metric else None

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

            if self.callback:
                self.callback.start_batch(
                    self._create_args_for_callback(batch_index=index_batch,
                                                   batch_size=BATCH_SIZE,
                                                   epoch=kwargs.get('epoch')))

            # move batch and targets to device
            logger.debug("Move X and y to the selected device")
            X, y = self._move_to_device(X, y)

            # forward the input in the correct way:
            # e.g:
            # forward(input)
            # forward(input1,input2,input3) we have to unfold the list or the tuple
            logger.debug("Forwarding the input")
            out = self._forward(X)
            Trainer._assert_out_requires_grad(out, train=True)

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

            if self.callback:
                self.callback.end_batch(
                    self._create_args_for_callback(batch_index=index_batch,
                                                   batch_size=BATCH_SIZE,
                                                   epoch=kwargs.get('epoch'),
                                                   batch_metrics=self._return_training_state(epoch_loss, epoch_metric)))

        # average the loss
        epoch_loss = epoch_loss / total_samples

        # average the metric
        if self.metric:
            epoch_metric = Trainer._execute_operation(operator.__truediv__, epoch_metric, len(loader))

        # return the averaged loss in the whole dataset and metrics
        logger.debug("Returning loss and metrics of the whole dataset")
        return self._return_training_state(epoch_loss=epoch_loss, epoch_metric=epoch_metric)

    def validation(self, loader: torch.utils.data.DataLoader, **kwargs) -> 'TrainingState':
        """Define the validation loop for one epoch

        Args:
            loader: DataLoader for extracting validation batches

        Returns:
            TrainingState (a namedtuple generated dynamically) which contains loss and metric(s) (if self.metric is not None)

        See Also
            :class:`~Trainer._register_training_state_type(metric)` for understanding the return type
        """
        self.model.eval()

        epoch_loss = 0

        total_samples = 0

        epoch_metric = Trainer._init_metric_obj(self.metric) if self.metric else None

        for index_batch, (X, y) in enumerate(tqdm(loader, total=len(loader), unit="batch", leave=False)):
            BATCH_SIZE = self._get_batch_size(X)
            total_samples += BATCH_SIZE

            if self.callback:
                self.callback.start_batch(
                    **self._create_args_for_callback(batch_index=index_batch,
                                                     batch_size=BATCH_SIZE,
                                                     epoch=kwargs.get('epoch')))

            X, y = self._move_to_device(X, y)

            out = self._forward(X)
            Trainer._assert_out_requires_grad(out, train=False)

            assert out.size() == y.size()
            loss = self.criterion(out, y)

            epoch_loss += loss.item() * BATCH_SIZE

            if self.metric:
                computed_metric = Trainer._compute_metric(self.metric, prediction=out, target=y)
                epoch_metric = Trainer._execute_operation(operator.add, epoch_metric, computed_metric)

            if self.callback:
                self.callback.end_batch(
                    **self._create_args_for_callback(batch_index=index_batch,
                                                     batch_size=BATCH_SIZE,
                                                     epoch=kwargs.get('epoch'),
                                                     batch_metrics=self._return_training_state(epoch_loss,
                                                                                               epoch_metric)))

        epoch_loss = epoch_loss / total_samples

        if self.metric:
            epoch_metric = Trainer._execute_operation(operator.__truediv__, epoch_metric, len(loader))

        return self._return_training_state(epoch_loss=epoch_loss, epoch_metric=epoch_metric)

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
            epochs: int) -> 'HistoryState':
        """Define the train-validation loop over all epochs

        Args:
            train_loader: DataLoader for extracting train batches
            val_loader: DataLoader for extracting validation batches
            epochs: amount of epochs

        Returns:
            HistoryState (a namedtuple that depends on TrainingState) which contains two lists:
                - train: List[TrainingState]
                - val  : List[TrainingState]

        See Also
            :class:`~Trainer._register_training_state_type(metric)` for understanding TrainingState
            :class:`~Trainer._register_history_state_type()` for understanding HistoryState
        """

        # RESUMING ATTRIBUTES
        # since the training is stopped at the beginning of the loop the last_epoch is indeed the new epoch to do
        # after resuming the object Trainer
        resuming = self._init_param_for_resuming('resuming', default=False)
        start_epoch = self._init_param_for_resuming('last_epoch', default=1)
        last_loss = self._init_param_for_resuming('last_loss', default=None)
        history = self._init_param_for_resuming('last_history', self.HistoryState(list(), list()))
        logger.debug(f"Is resuming? {resuming}")
        ###

        for epoch in range(start_epoch, epochs + 1):

            if self.stop:
                # TODO Save experiment
                break

            if self.callback:
                self.callback.start_epoch(**self._create_args_for_callback(resuming=resuming,
                                                                           epoch=epoch,
                                                                           last_loss=last_loss)
                                          )

            train_state = self.train(train_loader, epoch=epoch)
            val_state = self.validation(val_loader, epoch=epoch)

            history.train.append(train_state)
            history.val.append(val_state)

            # logger.debug(Trainer._return_loss_and_metric_formatted(train_state, train=True))
            # logger.debug(Trainer._return_loss_and_metric_formatted(val_state, train=False))

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if self.callback:
                self.callback.end_epoch(**self._create_args_for_callback(resuming=resuming,
                                                                         epoch=epoch,
                                                                         train_state=train_state,
                                                                         val_state=val_state,
                                                                         current_history=history)
                                        )

        return history

    def _init_param_for_resuming(self, key, default):
        """Define a function for initialize parameters for resuming

        Args:
            key: Parameter to check
            default: Default value to assign

        Returns:
            Default or saved value
        """
        return default if not hasattr(self, key) else getattr(self, key)

    @staticmethod
    def _register_training_state_type(metric: Union[Metric, List[Metric], Dict[str, Metric]]) -> 'TrainingState':
        """Register the TrainingState type defined at runtime

        Args:
            metric: Metric object. (Can be: Metric, List[Metric], Dict[str,Metric])

        Returns:
            TrainingState namedtuple:
                TrainingState(loss, metric_1, metric_2,...,metric_n)
        """
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

        TrainingState = namedtuple("TrainingState", field_names)

        # register in globals in order to be visible for pickling
        globals()[TrainingState.__name__] = TrainingState
        return TrainingState

    def _register_history_state_type(self) -> 'HistoryState':
        """Register the HistoryState type defined at runtime depending on TrainingState

        Returns:
            HistoryState namedtuple:
            HistoryState(train=List[TrainingState], val=List[TrainingState])
        """
        TrainingState = self.TrainingState

        HistoryState = NamedTuple("HistoryState", [
            ("train", List[TrainingState]),
            ("val", List[TrainingState])
        ])
        # register in globals in order to be visible for pickling
        globals()[HistoryState.__name__] = HistoryState
        return HistoryState

    def _return_training_state(self, epoch_loss: float,
                               epoch_metric: Union[Metric, List[Metric], Dict[str, Metric]]) -> 'TrainingState':
        """Utility function for creating a TrainingState object given the inputs

        Args:
            epoch_loss: the computed loss
            epoch_metric: the computed metric(s)

        Returns:
            TrainingState (namedtuple) object. (see :class:`~Trainer._register_training_state_type(metric)` for understanding the returned type)
        """
        if self.metric:
            if isinstance(epoch_metric, Dict):
                epoch_metric = epoch_metric.values()

            if isinstance(epoch_metric, float):
                return self.TrainingState(epoch_loss, epoch_metric)
            else:
                # epoch_metric is a list
                return self.TrainingState(epoch_loss, *epoch_metric)
        else:
            return self.TrainingState(epoch_loss)

    @staticmethod
    def _return_loss_and_metric_formatted(state: 'TrainingState', train: bool) -> str:
        """Format TrainingState object in  readable way for printing
        Args:
            state: TrainingState to format
            train: if the trainer is training or validating

        Returns:
            Formatted TrainingState as string
        """
        split = "train" if train else "val"
        return " ".join(
            f"{split}_{key}: {value}" for key, value in state._asdict().items()
        )

    @staticmethod
    def _validate_field_names(name: str):
        """Validate names in order to be accepted as field name for a namedtuple

        Args:
            name: name to rename

        Returns:
            well foramtted name
        """
        return re.sub(r"[^\w\d]+", '_', name)

    @staticmethod
    def _get_metric_name(metric: Union[Metric, List[Metric], Dict[str, Metric]]):
        """Returns a name or list of names for each metric

        Args:
            metric: metric(s) to extract the name

        Returns:
            name or list of names
        """
        if isinstance(metric, torch.nn.Module):
            return str(metric.__class__.__name__).lower().strip()
        elif isinstance(metric, Callable):
            return str(metric.__name__).lower().strip()
        elif isinstance(metric, List):
            return [Trainer._get_metric_name(metric_fn) for metric_fn in metric]
        elif isinstance(metric, Dict):
            return [key.lower().strip() for key in metric]

    def _forward(self, X):
        """Define the base forward

        Args:
            X: input

        Returns:
            output
        """
        if isinstance(X, torch.Tensor):
            return self.model.train_step(X) if self.model.training else self.model.val_step(X)
        elif isinstance(X, (List, Tuple)):
            return self.model.train_step(*X) if self.model.training else self.model.val_step(*X)
        else:
            raise RuntimeError(f"Invalid type: {type(X)}")

    @staticmethod
    def _init_metric_obj(metric: Union[Metric, List[Metric], Dict[str, Metric]]) -> Union[
        float, List[float], Dict[str, float]]:
        """Initialize the metric(s)

        Args:
            metric:

        Returns:
            - float value if is Metric
            - list of float if it is a list of Metrics
            - dict of float if it is a dict of Metrics
        """
        # if isinstance(self.metric, Metric):
        # It's a python design decision to not use GenericAlias
        if isinstance(metric, (Callable, torch.nn.Module)):
            logger.debug("metric is a function")
            return 0.
        elif isinstance(metric, List):
            logger.debug("metric is a list")
            return [0. for _ in range(len(metric))]
        elif isinstance(metric, Dict):
            logger.debug("metric is a dict")
            return {key: 0. for key in metric}
        else:
            raise RuntimeError(
                f"Invalid type: received {type(metric)}, expected: Metric,List[Metric],Dict[str,Metric]")

    @staticmethod
    def _compute_metric(metric: Union[Metric, List[Metric], Dict[str, Metric]], prediction: torch.Tensor,
                        target: torch.Tensor) -> [float, List[float], Dict[str, float]]:
        """
        Compute metric on prediction and target
        Args:
            metric: metric(s) to use
            prediction: output of the network
            target: target to compare

        Returns:
            - float if metric is Metric
            - list of float if metric is a List[Metric]
            - dict of float if metric is a Dict[str,Metric]
        """
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
            # TODO: Allow metric with more than one input and target
            raise RuntimeError(
                f"Unsupported type: (prediction){type(prediction)}, (target){type(target)}, expected Tensor")

    @staticmethod
    def _assert_out_requires_grad(out, train) -> NoReturn:
        """Assert that out contains or doesn't contains gradient based on if trainer is training or validating

        Args:
            out: output of the train_step or val_step
            train: define if the Trainer is training or validating
        """
        if isinstance(out, torch.Tensor):
            assert out.requires_grad == train
        elif isinstance(out, (List, Tuple)):
            for elem in out:
                if isinstance(elem, torch.Tensor):
                    assert elem.requires_grad == train

    def _get_batch_size(self, X) -> int:
        """Get the size of a batch from the input tensor X

        Args:
            X: batch tensor given by the DataLoader

        Returns:
            Size of a batch
        """
        if isinstance(X, torch.Tensor):
            return X.size(0)
        elif isinstance(X, (List, Tuple)):
            sizes = [elem.size(0) for elem in X if isinstance(elem, torch.Tensor)]
            return sizes[0]
        else:
            raise RuntimeError(f"Invalid type: {type(X)}")

    def _move_to_device(self, X, y) -> NoReturn:
        """ Move input and target to device

        Args:
            X: batch input returned by the DataLoader
            y: batch target returned by the DataLoader
        """
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

    def _create_args_for_callback(self, **kwargs) -> Dict:
        """ Return a dictionary of arguments to pass to callbacks

        Args:
            **kwargs: named arguments to pass

        Returns:
            {
                "trainer" : self
                **kwargs
        """

        base_kwargs = {
            "stop_fn": self._stop_fn,
            "log_dir": self.log_dir,
            "optimizer": self.optimizer,
            "train": self.model.training
        }

        return {**base_kwargs, **kwargs}

    @staticmethod
    def _execute_operation(operation, first, second):
        """ Execute operation between first and second (handling operation element wise if they are List or Dict)
        Args:
            operation: scalar operation to do (e.g. add)
            first: left element
            second: right element

        Returns:
            output of the operation
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
            # in case second is a int or float, create a list with second repeated as much as len(first)
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


    #TODO Rwrite save_model, save_experiment, load_experiment

    def save_model(self) -> NoReturn:
        """ Function for save the parameters and state dictionary of the model
        """
        try:
            dict_to_save = {"param": self.model.__dict__,
                            "state": self.model.state_dict()
                            }
            model_path = self.model_info.model_dir / f"{self.model_info.name}_state_dict.pytorch"
            torch.save(dict_to_save, model_path)
        except:
            logger.error("An error occurred saving the model.")

    def save_experiment(self, last_epoch: int, last_loss: float, current_history: 'HistoryState') -> NoReturn:
        """ A function for saving the Trainer for resuming the training

        Args:
            last_epoch: last epoch done ( since the trainer is stopped at the beginning of the loop the last epoch is the epoch from which to resume the training )
            last_loss: last computed loss (for init early stopping )
            current_history: last computed history ( in order to merge the old history with new one )
        """
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
                #"callback": self.callback,
                "device": self.device,
                #
                "epoch": last_epoch,
                "loss": last_loss,
                #
                "current_history": current_history
            }

            torch.save(what_to_save, checkpoint_path)
            logger.debug("Information saved.")

        except AttributeError as ae:
            if "Can't pickle local object":
                new_message = str(ae) + "\n"
                new_message += "Insert the unpickable function inside a separate module and import the function."
                logger.error(new_message)
                ae = RuntimeError(new_message)
            raise ae

    @classmethod
    def load_experiment(cls, base_model_dir: str, experiment: str, last_epoch: int):
        """Load function for initialize Trainer object given the state sved before

        Args:
            base_model_dir: dir of the model to load
            experiment: experiment to load
            last_epoch: last epoch to load

        Returns:

        """

        checkpoint_path = utility.get_models_dir() / base_model_dir / experiment / f"trainer_{last_epoch}.pytorch"
        if not checkpoint_path.exists():
            raise RuntimeError(f"{checkpoint_path} doesn't exists.")

        what_to_load = torch.load(checkpoint_path)
        logger.debug(what_to_load)
        obj = cls(experiment=what_to_load['experiment'],
                  model=what_to_load['model'],
                  optimizer=what_to_load['optimizer'],
                  criterion=what_to_load['criterion'],
                  metric=what_to_load['metric'],
                  lr_scheduler=what_to_load['lr_scheduler'],
                  #callback=what_to_load['callback'],
                  device=what_to_load['device'])

        # for resuming training
        obj.resuming = True
        obj.last_epoch = what_to_load['epoch']
        obj.last_loss = what_to_load['loss']
        obj.last_history = what_to_load['current_history']

        return obj
