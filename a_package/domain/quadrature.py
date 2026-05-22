from typing import Sequence

import numpy as np


from .field import Field, field_element_axs


class Quadrature:
    """Quadrature for numerical approximating an integral.

    The implementation assumes a regular grid.
    """

    def __init__(self, quad_pt_coords: Sequence[Sequence[float]], quad_pt_weights: Sequence[float]):
        if np.shape(quad_pt_coords)[0] != np.size(quad_pt_weights):
            raise ValueError("Number of quadrature points must match the number of weights")
        if not np.isclose(np.sum(quad_pt_weights), 1.):
            raise ValueError("Quadrature weights must sum to 1")
        self.quad_pt_coords = np.asarray(quad_pt_coords)
        self.quad_pt_weights = np.asarray(quad_pt_weights)

    @property
    def nb_quad_pts(self):
        return np.size(self.quad_pt_weights)

    def integrate(self, field: Field, element_area: float=1.0):
        # Due to regular grid, it is possible to factor out the element area
        element_sum = element_area * np.sum(field, axis=field_element_axs)
        return np.einsum("s, cs-> c", self.quad_pt_weights, element_sum)

    def propag_integral_weight(self, field: Field, element_area: float=1.0):
        return element_area * np.einsum("s, cs...-> cs...", self.quad_pt_weights, field)


centroid_quadrature = Quadrature([[1 / 3, 1 / 3], [2 / 3, 2 / 3]], [0.5, 0.5])
"""Numerical quadrature with points located at the centroid of the two triangular elements of each pixel."""

three_pt_quadrature = Quadrature(
    [[4 / 6, 1 / 6], [1 / 6, 1 / 6], [1 / 6, 4 / 6], [2 / 6, 5 / 6], [5 / 6, 5 / 6], [5 / 6, 2 / 6]],
    [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
