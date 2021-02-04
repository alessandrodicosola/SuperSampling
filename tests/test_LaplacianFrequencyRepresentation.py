from unittest import TestCase

from models.LaplacianFrequencyRepresentation import LaplacianFrequencyRepresentation

lfr = LaplacianFrequencyRepresentation(1, 2, 11)


class TestLaplacianFrequencyRepresentation(TestCase):
    def test_get_recursions(self):
        self.assertEqual(lfr.get_recursions(2), 1)

        self.assertEqual(lfr.get_recursions(3.5), 2)

        self.assertEqual(lfr.get_recursions(9.3),4)
