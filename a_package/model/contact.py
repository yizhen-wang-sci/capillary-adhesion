import numpy as np

from a_package.domain import adapt_shape


class RigidContact:
    """Computes the gap field between two rigid surfaces at a given separation."""

    def __init__(self, upper: np.ndarray, lower: np.ndarray):
        self.upper = adapt_shape(upper)
        self.lower = adapt_shape(lower)

    def set_mean_separation(self, value: float):
        self.separation = value

    def get_gap(self):
        return np.clip(self.separation + self.upper - self.lower, 0, None)
