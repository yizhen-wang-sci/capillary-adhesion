"""Field data as a 4D NumPy array, and the convention for its dimensions."""

from typing import TypeAlias

import numpy as np

Field: TypeAlias = np.ndarray[tuple[int, int, int, int]]
"""A 4D array holding field data."""
field_component_ax = 0
"""Axis of the field component."""
field_sub_pt_ax = 1
"""Axis of the sub-point within an element."""
field_element_axs = (2, 3)
"""The two axes spanning the elements."""


def adapt_shape(array: np.ndarray) -> Field:
    """Insert the missing leading axes so an array satisfies the field convention.

    Args:
        array: 2D, holding the element axes alone; or 4D, following the convention.

    Returns:
        A 4D view of a 2D input; or a 4D input, unchanged.

    Raises:
        ValueError: If the array is neither 2D nor 4D.
    """
    match(np.ndim(array)):
        case 2:
            return np.expand_dims(array, axis=(field_component_ax, field_sub_pt_ax))
        case 4:
            return array
        case _:
            raise ValueError(f"Array of {np.ndim(array)}D, expect 2D or 4D")
