from pathlib import Path

from pathlib import Path

def raise_if_path_doesnt_exist(path:Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return path


def get_project_dir() -> Path:
    """
    :return: Directory root of the project ( where **src**, **tests**, **datasets**, ... are.
    """
    dir = Path(__file__).parent.parent
    return raise_if_path_doesnt_exist(dir)

def get_datasets_dir() -> Path:
    """
    :return: Directory that contains all datasets
    """
    dir = get_project_dir() / "datasets"
    return raise_if_path_doesnt_exist(dir)

def get_models_dir() -> Path:
    """
    :return: Directory where trained models are saved
    """
    return get_project_dir() / "models"
