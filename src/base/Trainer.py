import torch
import torch.optim
import torch.utils.data
import torch.utils.tensorboard

from base import BaseModel
from base.Callback import Callback, ListCallback

from base.hints import Union, List, Tuple

from tqdm.auto import tqdm

import sys
import logging

logging.basicConfig()


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
                 criterion,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 callback: Union[Callback, ListCallback] = None,
                 device : torch.device = None):

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.experiment = experiment

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.callback = callback

        # flag used for stopping the training
        self.stop = False

        # define where tensorboard events will be saved
        tensorboard_dir = self.model.model_dir / experiment
        self.tensorboard = torch.utils.tensorboard.SummaryWriter(str(tensorboard_dir))

        # define the logger
        # log = self.model.model_dir / "logs" / f"{experiment}.log"
        # self.logger = logging.getLogger(f"{__name__}_{model.name}")
        # self.logger.setLevel(logging.INFO)
        # consoleHandler = logging.StreamHandler(sys.stdout)
        # fileHandler = logging.FileHandler(filename=log, mode="w")
        # self.logger.addHandler(consoleHandler)
        # self.logger.addHandler(fileHandler)

        # self.logger.info(f"Initialized Trainer for {experiment} with {model.name}")

    def stop(self):
        self.stop = True

    def train(self, loader: torch.utils.data.DataLoader):
        """Define the train loop for one epoch"""

        self.model.train()

        epoch_loss = 0

        for index_batch, (X, y) in enumerate(tqdm(loader, total=len(loader), unit="batch", leave=False)):
            #extract batch size from X
            BATCH_SIZE = self._get_batch_size(X)

            # move batch and targets to device
            X, y = self._convert_to_device(X, y)

            # erase all saved gradients
            self.optimizer.zero_grad()

            # forward the input in the correct way:
            # e.g:
            # forward(input)
            # forward(input1,input2,input3) we have to unfold the list or the tuple
            if isinstance(X, torch.Tensor):
                out = self.model.train_step(X)
            elif isinstance(X, (List, Tuple)):
                out = self.model.train_step(*X)
            else:
                raise RuntimeError(f"Invalid type: {type(X)}")

            # compute the loss
            loss = self.criterion(out, y)
            # compute the gradients
            loss.backward()
            # update the parameters
            self.optimizer.step()

            # update the loss
            # loss.item() contains the loss averaged in the batch
            # * X.size(0) we get the non averaged loss in the batch
            epoch_loss += loss.item() * BATCH_SIZE

        # return the averaged loss in the whole dataset
        return epoch_loss / len(loader)

    def validation(self, loader: torch.utils.data.DataLoader):
        self.model.eval()

        epoch_loss = 0

        with torch.no_grad():
            for index_batch, (X, y) in enumerate(tqdm(loader, total=len(loader), unit="batch", leave=False)):
                BATCH_SIZE = self._get_batch_size(X)

                X, y = self._convert_to_device(X,y)

                if isinstance(X, torch.Tensor):
                    out = self.model.val_step(X)
                elif isinstance(X, (List, Tuple)):
                    out = self.model.val_step(*X)
                else:
                    raise RuntimeError(f"Invalid type: {type(X)}")

                loss = self.criterion(out, y)

                epoch_loss += loss.item() * BATCH_SIZE

        return epoch_loss / len(loader)

    def fit(self, train_loader : torch.utils.data.DataLoader, val_loader : torch.utils.data.DataLoader, epochs: int):

        for epoch in range(1, epochs + 1):

            if self.stop:
                logging.info("Stopping training.")
                break

            if self.callback:
                self.callback.start_epoch(self._create_args_for_callback(epoch = epoch))

            train_loss = self.train(train_loader)
            val_loss   = self.validation(val_loader)

            print(f"Epoch: {epoch} Train Loss: {train_loss} Val Loss: {val_loss}")

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if self.callback:
                self.callback.end_epoch(self._create_args_for_callback(train_loss = train_loss, val_loss = val_loss, epoch = epoch))


    def _get_batch_size(self,X):
        if isinstance(X, torch.Tensor):
            return X.size(0)
        elif isinstance(X, (List,Tuple)):
            sizes = [elem.size(0) for elem in X if isinstance(elem,torch.Tensor)]
            return sizes[0]
        else:
            raise RuntimeError(f"Invalid type: {type(X)}")


    def _convert_to_device(self,X,y):
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

        return X,y


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
            "experiment" : self.experiment,
            "model" : self.model,
            "optimizer": self.optimizer,
        }

        return {**base_kwargs, **kwargs}