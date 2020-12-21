import unittest
from abc import abstractmethod, ABC

import torch


class PyTorchTest(unittest.TestCase, ABC):

    def setUp(self) -> None:
        self._before()

    def tearDown(self) -> None:
        self._after()

    def _before(self):
        torch.manual_seed(2020)
        torch.cuda.manual_seed(2020)
        torch.set_deterministic(True)

        self.before()

    def _after(self):
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()

        self.after()

    @abstractmethod
    def before(self):
        pass

    @abstractmethod
    def after(self):
        pass
