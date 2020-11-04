from unittest import TestCase
import utility
from pathlib import Path
import os

class Test(TestCase):
    _test_folder = Path(os.getcwd())
    def test_get_project_dir(self):
        true_project_folder = self._test_folder.parent
        self.assertEqual(true_project_folder,utility.get_project_dir())

    def test_get_datasets_dir(self):
        true_datasets_folder = utility.get_project_dir() / "datasets"
        self.assertEqual(true_datasets_folder,utility.get_datasets_dir())

    def test_get_models_dir(self):
        true_models_folder = utility.get_project_dir() / "models"
        self.assertEqual(true_models_folder,utility.get_models_dir())
