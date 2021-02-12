from typing import Tuple, List

import torch

from base import BaseModule


def detach(device, X):
    if isinstance(X, torch.Tensor):
        return X.detach().to(device)
    elif isinstance(X, (Tuple, List)):
        return [x.detach().to(device) if isinstance(x, torch.Tensor) else x for x in X]
    else:
        raise RuntimeError(
            f"Invalid type for X. received: {type(X)}, expected: Tensor, Tuple[Tensor], List[Tensor].")


def param_in_function(function, *params):
    from inspect import signature
    parameters = signature(function).parameters
    flag = True
    for param in params:
        flag = flag and (param in parameters)

    return flag


def get_batch_size(X) -> int:
    """Get the size of a batch from the input tensor X

    Args:
        X: batch tensor given by the DataLoader

    Returns:
        Size of a batch

    Raises:
        RuntimeError if X is an invalid type, expected: Tensor,List,Tuple
    """
    if isinstance(X, torch.Tensor):
        return X.size(0)
    elif isinstance(X, (List, Tuple)):
        sizes = [elem.size(0) for elem in X if isinstance(elem, torch.Tensor)]
        return sizes[0]
    else:
        raise RuntimeError(f"Invalid type: {type(X)}")


def assert_tensor_requires_grad(tensor, train: bool):
    """Assert that out contains or doesn't contains gradient based on if trainer is training or validating

    Args:
        out: output of the train_step or val_step
        train: define if the Trainer is training or validating

    Raises:
        AssertError if requires_grad != train
    """
    if isinstance(tensor, torch.Tensor):
        assert tensor.requires_grad == train
    elif isinstance(tensor, (List, Tuple)):
        for elem in tensor:
            if isinstance(elem, torch.Tensor):
                assert elem.requires_grad == train


def move_to_device(X, y, device):
    """ Move input and target to device

    Args:
        X: batch input returned by the DataLoader
        y: batch target returned by the DataLoader

    Raises:
        RuntimeError if X is an invalid type
    """
    if isinstance(X, torch.Tensor):
        X = X.to(device)
    elif isinstance(X, Tuple):
        X = tuple([elem.to(device) if isinstance(elem, torch.Tensor) else elem for elem in X])
    elif isinstance(X, List):
        X = [elem.to(device) if isinstance(elem, torch.Tensor) else elem for elem in X]
    else:
        raise RuntimeError(f"Invalid type: {type(X)}")

    if isinstance(y, torch.Tensor):
        y = y.to(device)
    elif isinstance(X, Tuple):
        y = tuple([elem.to(device) if isinstance(elem, torch.Tensor) else elem for elem in y])
    elif isinstance(X, List):
        y = [elem.to(device) if isinstance(elem, torch.Tensor) else elem for elem in y]
    else:
        raise RuntimeError(f"Invalid type: {type(y)}")

    return X, y

def forward(model : BaseModule, X):
    """Define the base forward

    Args:
        X: input

    Returns:
        output

    Raises:
        RuntimeError if X is an invalid type, expected: Tensor,List,Tuple
    """
    if isinstance(X, torch.Tensor):
        return model.train_step(X) if model.training else model.val_step(X)
    elif isinstance(X, (List, Tuple)):
        return model.train_step(*X) if model.training else model.val_step(*X)
    else:
        raise RuntimeError(f"Invalid type: {type(X)}")