import unittest
from pathlib import Path
from typing import Tuple
from unittest import TestCase
from base import BaseModule
import torch


class BaseModuleOneInput(BaseModule):
    def forward(self, input):
        # do some computation otherwise requires_grad is not set
        input = input + 0.0
        return input

    def train_step(self, input):
        return self(input)

    @torch.no_grad()
    def val_step(self, input):
        return self(input)

    @torch.no_grad()
    def test_step(self, input):
        return self(input)


class BaseModuleTwoInput(BaseModule):
    def forward(self, input1, input2):
        # do some computation otherwise requires_grad is not set
        input1 = input1 + 0.0
        input2 = input2 + 0.0

        return input1 + input2

    def train_step(self, input1, input2):
        return self(input1, input2)

    @torch.no_grad()
    def val_step(self, input1, input2):
        return self(input1, input2)

    @torch.no_grad()
    def test_step(self, input1, input2):
        return self(input1, input2)


class BaseModuleTwoOutput(BaseModule):
    def forward(self, input):
        # do some computation otherwise requires_grad is not set
        input = input + 0.0
        return (input, input)

    def train_step(self, input):
        return self(input)

    @torch.no_grad()
    def val_step(self, input):
        return self(input)

    @torch.no_grad()
    def test_step(self, input):
        return self(input)


class TestBaseModel(TestCase):

    def test_one_input_grad(self):
        input = torch.rand(5, requires_grad=True)
        model = BaseModuleOneInput()

        model.train()
        self.assertTrue(model(input).requires_grad)
        self.assertTrue(model.train_step(input).requires_grad)

        model.eval()
        self.assertFalse(model.val_step(input).requires_grad)
        self.assertFalse(model.test_step(input).requires_grad)

    def test_two_input_grad(self):
        input = torch.rand(5, requires_grad=True)
        model = BaseModuleTwoInput()

        model.train()
        self.assertTrue(model(input, input).requires_grad)
        self.assertTrue(model.train_step(input, input).requires_grad)

        model.eval()
        self.assertFalse(model.val_step(input, input).requires_grad)
        self.assertFalse(model.test_step(input, input).requires_grad)

    def test_two_output_grad(self):
        input = torch.rand(5, requires_grad=True)
        model = BaseModuleTwoOutput()

        model.train()
        out = model(input)
        self.assertIsInstance(out, Tuple)
        for elem in out:
            self.assertTrue(elem.requires_grad)

        out = model.train_step(input)
        self.assertIsInstance(out, Tuple)
        for elem in out:
            self.assertTrue(elem.requires_grad)

        model.eval()
        out = model.val_step(input)
        self.assertIsInstance(out, Tuple)
        for elem in out:
            self.assertFalse(elem.requires_grad)

        out = model.test_step(input)
        self.assertIsInstance(out, Tuple)
        for elem in out:
            self.assertFalse(elem.requires_grad)

if __name__ == "__main__":
    unittest.main()
