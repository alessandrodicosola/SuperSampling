import inspect
import unittest
from abc import abstractmethod, ABC
from collections import Callable
from inspect import signature

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from base.Callbacks import Callback
from base.hints import Metric, Criterion


class PyTorchTest(unittest.TestCase, ABC):

    def setUp(self) -> None:
        self.addTypeEqualityFunc(Callback, self.Callback_are_equal)
        self.addTypeEqualityFunc(Metric, self.Metric_are_equal)
        self.addTypeEqualityFunc(Criterion, self.Criterion_are_equal)

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

    # PyTorch comparing function
    def Callback_are_equal(self, cls1: Callback, cls2: Callback, msg=None):
        keys1 = signature(cls1.__init__).parameters.keys()
        keys2 = signature(cls2.__init__).parameters.keys()

        for key1, key2 in zip(keys1, keys2):
            self.assertEqual(key1, key2)
            if key1 in cls1.__dict__:
                self.assertTrue(key2 in cls2.__dict__)
                self.assertEqual(cls1.__dict__[key1], cls2.__dict__[key2])

    def Metric_are_equal(self, m1: Metric, m2: Metric, msg=None):
        if isinstance(m1, Module):
            keys1 = signature(m1.__init__).parameters.keys()
            keys2 = signature(m2.__init__).parameters.keys()
        elif isinstance(m1, Callable):
            keys1 = signature(m1).parameters.keys()
            keys2 = signature(m2).parameters.keys()
        else:
            raise NotImplementedError(type(m1))

        for key1, key2 in zip(keys1, keys2):
            self.assertEqual(key1, key2)
            self.assertEqual(m1.__dict__[key1], m2.__dict__[key2])

    def Criterion_are_equal(self, c1: Criterion, c2: Criterion, msg=None):
        keys1 = signature(c1.__init__).parameters.keys()
        keys2 = signature(c2.__init__).parameters.keys()

        for key1, key2 in zip(keys1, keys2):
            self.assertEqual(key1, key2)
            if key1 in c1.__dict__:
                self.assertTrue(key2 in c2.__dict__)
                self.assertEqual(c1.__dict__[key1], c2.__dict__[key2])

    def Tensors_are_equal(self, t1: Tensor, t2: Tensor, msg=None):
        return t1.eq(t2).all()

    def Tensors_are_not_equal(self, t1: Tensor, t2: Tensor):
        return not self.Tensors_are_equal(t1, t2)

    def Tensors_are_similar(self, t1: Tensor, t2: Tensor):
        return torch.isclose(t1, t2)

    def Tensors_are_not_similar(self, t1: Tensor, t2: Tensor):
        return not self.Tensors_are_similar(t1, t2)

    def dict_are_equal(self, d1: dict, d2: dict):
        for (key1, value1), (key2, value2) in zip(d1.items(), d2.items()):
            self.assertEqual(key1, key2)
            if isinstance(value1, Tensor):
                self.assertIsInstance(value2, Tensor)
                self.Tensors_are_equal(value1, value2)
            elif isinstance(value1, dict):
                self.assertIsInstance(value2, dict)
                self.dict_are_equal(value1, value2)
            else:
                self.assertEqual(value1, value2)

    state_dicts_are_equal = dict_are_equal

    def cls_are_equal(self, cls1, cls2):
        self.assertIsInstance(cls1, type(cls2))
        self.assertIsInstance(cls2, type(cls1))
        if isinstance(cls1, Criterion):
            self.assertIsInstance(cls2, Criterion)
            self.Criterion_are_equal(cls1, cls2)
        elif isinstance(cls1, Callback):
            self.assertIsInstance(cls2, Callback)
            self.Callback_are_equal(cls1, cls2)
        else:
            self.assertEqual(cls1, cls2)
