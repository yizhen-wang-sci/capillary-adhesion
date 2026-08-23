"""Contact between surfaces."""

import numpy as np

from a_package.domain import adapt_shape


class RigidContact:
    """Computes the gap field between two rigid surfaces at a given separation."""

    def __init__(self, upper: np.ndarray, lower: np.ndarray):
        """Store the two surface profiles, shaped to the field convention.

        Args:
            upper: Height profile of the upper surface. Passed through `adapt_shape`, so either
                the bare grid shape or the full field shape will do.
            lower: Height profile of the lower surface, same shapes accepted.
        """
        self.upper = adapt_shape(upper)
        self.lower = adapt_shape(lower)

    def set_mean_separation(self, value: float):
        """Set the mean separation the gap is measured at.

        Args:
            value: The mean separation.
        """
        self.separation = value

    def get_gap(self):
        """Gap between the two surfaces at the separation set by `set_mean_separation`.

        Returns:
            The gap, zeroed wherever the surfaces would collide.
        """
        return np.clip(self.separation + self.upper - self.lower, 0, None)
