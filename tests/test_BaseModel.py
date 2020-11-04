from unittest import TestCase
from base import BaseModel

class TestBaseModel(TestCase):
    class TestBaseModel(BaseModel):

        def forward(self, input):
            pass

    def test_name(self):
        true_name = "TestBaseModel"
        model = self.TestBaseModel()
        self.assertEqual(true_name,model.name)
