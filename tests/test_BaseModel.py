from unittest import TestCase
from base import BaseModel
import torch
class TestBaseModel(TestCase):
    class TestBaseModel(BaseModel):
        def __init__(self, hidden_units):
            super().__init__()
            self.linear = torch.nn.Linear(in_features=hidden_units,out_features=hidden_units)
            self.relu = torch.nn.ReLU()

        def train_step(self, batch, batch_index):
            super().train_step(batch,batch_index)


        def val_step(self, batch, batch_index):
            super().val_step(batch,batch_index)

        def forward(self, input):
            x = self.linear(input)
            x = self.relu(x)
            return x

    def test_name(self):
        true_name = "TestBaseModel"
        model = self.TestBaseModel(5)
        self.assertEqual(true_name,model.name)

    def test_forward_gradient_used(self):
        import torch
        model = self.TestBaseModel(5)
        out = model(torch.rand(5))
        self.assertTrue(out.requires_grad)

    def test_forward_no_gradient_used(self):
        import torch
        model = self.TestBaseModel(5)
        with torch.no_grad():
            out = model(torch.rand(5))
        self.assertFalse(out.requires_grad)

    def test_model_in_training(self):
        import torch
        model = self.TestBaseModel(5)
        model.train_step(None,0)
        self.assertTrue(model.training)

    def test_model_not_in_training(self):
        import torch
        model = self.TestBaseModel(5)
        model.val_step(None,0)
        self.assertFalse(model.training)
