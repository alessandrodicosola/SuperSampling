from collections import namedtuple
import math

# Represents a level in the laplacian pyramid
Level = namedtuple("Level", "index scale")


class LaplacianFrequencyRepresentation:
    """Define the Laplacian Frequency Representation

    Args:
        start: The first scale of the scale ranges
        end: The last scale of the scale ranges
        count: Number of pyramid level
    """

    def __init__(self, start: int, end: int, count: int):
        self.start = start
        self.end = end
        self.count = count

        step = (end - start) / (count - 1)
        scales = [start + l * step for l in range(count)]

        self.information = [Level(index, scale) for index, scale in enumerate(scales)]

    def get_index(self, scale: float):
        """ Return the corresponding pyramid index for the given decimal scale
        Args:
          scale: Decimal scale
        Returns:
          index
        """
        return math.ceil((self.count - 1) * (scale - 1))

    def get_weight(self, scale: float):
        """ Return the interpolation weight for the given decimal scale
        Args:
          scale: Decimal scale
        Returns:
          weight
        """
        return (self.count - 1) * (self.information[self.get_index(scale)].scale - scale)

    def get_for(self, scale: float):
        """Returns the i-th and (i-1)-th level of the pyramid given the decimal scale

        Args:
            scale: decimal scale
        """
        index = self.get_index(scale)
        return self.information[index - 1], self.information[index]

    def get_recursions(self, scale: float):
        return math.ceil(math.log2(scale))

    def get_scales(self, scale: float):
        n_recursions = self.get_recursions(scale)
        n_minus_1_recursions = n_recursions - 1
        if scale > self.information[-1].scale:
            r = scale / 2 ** n_minus_1_recursions
            return [self.information[-1].scale] * n_minus_1_recursions + [r]
        else:
            return [scale]
