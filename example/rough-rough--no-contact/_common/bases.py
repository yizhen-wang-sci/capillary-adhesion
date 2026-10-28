"""The frame a record's quantities are addressed in: two spatial axes and a step axis."""

from collections.abc import Sequence

import numpy as np

from a_package.dataset import QuantityFront
from a_package.domain import Grid, field_component_ax, field_sub_pt_ax

SPATIAL_BASES = ("x", "y")
"""The names of the bases NuMPI decomposes."""


def element_values(field: np.ndarray) -> np.ndarray:
    """The element values of a field, without its component and sub-point axes.

    A `RigidContact` gap comes back 4D, through `adapt_shape`, while a record stores it over
    ``x`` and ``y`` alone.

    Note:
        Not `np.squeeze`, which also drops a spatial axis where a decomposition leaves a
        subdomain one element wide.
    """
    assert field_component_ax == 0 and field_sub_pt_ax == 1
    return field[0, 0]


def save_grid(quantities: QuantityFront, names: Sequence[str], grid: Grid):
    """Save the coordinates of the grid's spatial axes, one basis per axis.

    A coordinate is the element's centre as a fraction of that axis's own length, so every axis
    runs over (0, 1).

    Args:
        quantities: The record's front. The named bases must already be defined on it.
        names: One basis name per axis, in the grid's axis order.
        grid: Where the pixel counts, the pixel sizes and the axis lengths come from.
    """
    for name, n, d, l in zip(names, grid.nb_domain_grid_pts, grid.element_sizes, grid.domain_lengths):
        quantities.save_value(name, (np.arange(n) + 0.5) * d / l)
