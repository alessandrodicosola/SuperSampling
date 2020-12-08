from base import BaseModel
import torch.optim

from .hints import Union, Path, NoReturn, OptimizerType


def save(model : BaseModel, current_epoch : int = None, current_loss : int = None) -> NoReturn:
    """
    Save a checkpoint of the model
    :param model: BaseModel to save
    :param current_epoch: ( 0-index epoch ). if not None **optimizer** and **epoch** are saved for resuming
    :param current_loss : current loss to save for resuing
    """

    resuming = current_epoch != None

    path = model.model_dir / f"{model.name}_checkpoint.pt"

    dict_to_save = dict()

    dict_to_save["resuming"] = resuming

    dict_to_save["model"] = dict()
    dict_to_save["model"]["state"] = model.state_dict()

    if resuming:
        if isinstance(model.optimizer, OptimizerType):
            # save the parameters of the optimizer
            dict_to_save["resuming"] = dict()
            dict_to_save["resuming"]["optimizer"] = model.optimizer.state_dict()

            # save the current epoch
            dict_to_save["resuming"]["epoch"] = current_epoch
            dict_to_save["resuming"]["loss"]  = current_loss
        else:
            raise RuntimeError("Unknown type")

    torch.save(dict_to_save,path)

def load(model : BaseModel, optimizer : OptimizerType):
    """
    Load model and if necessary information for resuming the training
    :param path:
    :param model:
    :param optimizer:
    :return:
    """
    path = model.model_dir / f"{model.name}"

    dict_to_load = torch.load(path)
    is_resuming = dict_to_load["resuming"]

    model.load_state_dict(dict_to_load["model"]["state"])

    if is_resuming:
        optimizer.load_state_dict(dict_to_load["resuming"]["optimizer"])
        epoch = dict_to_load["resuming"]["epoch"]
        loss = dict_to_load["resuming"]["loss"]
        return model,optimizer,epoch,loss

    return model
