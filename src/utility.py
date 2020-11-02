from pathlib import Path

from pathlib import Path

def raise_if_path_doesnt_exist(path:Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

def get_project_dir() -> Path:
    """
    :return: Directory root of the project ( where **src**, **tests**, **datasets**, ... are.
    """
    dir = Path(__file__).parent.parent
    raise_if_path_doesnt_exist(dir)
    return dir

def get_datasets_dir() -> Path:
    """
    :return: Directory that contains all datasets
    """
    dir = get_project_dir() / "datasets"
    raise_if_path_doesnt_exist(dir)
    return dir

def get_dataset_dir(name:str) -> Path:
    """
    :param name:
    :return: Directory where the specified dataset is saved (if it exist)
    """
    dir = get_datasets_dir() / name
    if not dir.exists():
        raise FileNotFoundError(f"{dir} doesn't exist")
    return dir

def get_models_dir() -> Path:
    """
    :return: Directory where trained models are saved
    """
    return get_project_dir() / "models"
