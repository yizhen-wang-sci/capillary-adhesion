"""Quadrature rules for integrating a field."""

import logging
from abc import ABC
from typing import ClassVar

import numpy as np
from NuMPI import MPI

from .field import Field, field_element_axs

logger = logging.getLogger(__name__)


class Quadrature(ABC):
    """An integral approximated as a weighted sum of values at the quadrature points."""

    quad_pt_coords: ClassVar[np.ndarray]
    """Coordinates of the quadrature points within a unit pixel, one row each. Read-only."""
    quad_pt_weights: ClassVar[np.ndarray]
    """Weight of each quadrature point, summing to 1. Read-only."""
    _REQUIRED = ("quad_pt_coords", "quad_pt_weights")

    def __init_subclass__(cls, **kwargs):
        """Validate and freeze a subclass's quadrature points and weights.

        Args:
            **kwargs: Forwarded to `super().__init_subclass__`.

        Raises:
            TypeError: If the subclass leaves `quad_pt_coords` or `quad_pt_weights` undefined.
            ValueError: If the coordinates are not 2-D, the weights not 1-D, their counts
                disagree, or the weights do not sum to 1.
        """
        super().__init_subclass__(**kwargs)

        # Ensure the subclass defines all required attributes
        missing = [name for name in cls._REQUIRED if name not in cls.__dict__]
        if missing:
            raise TypeError(f"{cls.__name__} must define class attribute(s): {', '.join(missing)}")

        # Value validation
        coords = np.asarray(cls.quad_pt_coords, dtype=float)
        weights = np.asarray(cls.quad_pt_weights, dtype=float)
        if coords.ndim != 2:
            raise ValueError(f"{cls.__name__}: quad_pt_coords must be 2-D, got shape {coords.shape}")
        if weights.ndim != 1:
            raise ValueError(f"{cls.__name__}: quad_pt_weights must be 1-D, got shape {weights.shape}")
        if coords.shape[0] != weights.size:
            raise ValueError(
                f"{cls.__name__}: number of quadrature points ({coords.shape[0]}) must "
                f"match number of weights ({weights.size})"
            )
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"{cls.__name__}: quadrature weights must sum to 1, got {weights.sum()}")

        # Freeze and store the validated arrays back to the class
        coords.flags.writeable = False
        weights.flags.writeable = False
        cls.quad_pt_coords = coords
        cls.quad_pt_weights = weights

    def __init__(self, communicator=MPI.COMM_SELF):
        """Bind the rule to a communicator.

        Args:
            communicator: Communicator across whose ranks `integrate` reduces, `MPI.COMM_SELF`
                by default.

        Raises:
            TypeError: If instantiated directly rather than through a subclass.
        """
        if type(self) is Quadrature:
            raise TypeError("Quadrature is abstract, instantiate a subclass instead.")
        self._communicator = communicator

    @property
    def nb_quad_pts(self):
        """Number of quadrature points per element."""
        return self.quad_pt_weights.size

    def integrate(self, field: Field, element_area: float = 1.0):
        """Integrate a field over the whole domain, across all ranks.

        Args:
            field: Values at the quadrature points.
            element_area: Area of one element.

        Returns:
            One integral per field component, reduced over the communicator.
        """
        # Regular grid -> element area factors out
        element_sum = element_area * np.sum(field, axis=field_element_axs)
        local = np.einsum("s, cs-> c", self.quad_pt_weights, element_sum)
        return self._communicator.allreduce(local, op=MPI.SUM)

    def propag_integral_weight(self, field: Field, element_area: float = 1.0):
        """Propagate the sensitivity of `integrate` back to the quadrature points.

        Args:
            field: Derivative of the integrand at each quadrature point.
            element_area: Area of one element.

        Returns:
            The derivative weighted per quadrature point, shaped like `field`, not reduced
            across ranks.
        """
        # Regular grid -> element area factors out
        return element_area * np.einsum("s, cs...-> cs...", self.quad_pt_weights, field)


class NodalQuadrature(Quadrature):
    """Quadrature by summing up nodal values."""

    quad_pt_coords: ClassVar[np.ndarray] = np.array([[0.0, 0.0]])
    quad_pt_weights: ClassVar[np.ndarray] = np.array([1.0])


class CentroidQuadrature(Quadrature):
    """Quadrature with two points, each located at the centroid of a triangular element."""

    quad_pt_coords: ClassVar[np.ndarray] = np.array([[1 / 3, 1 / 3], [2 / 3, 2 / 3]])
    quad_pt_weights: ClassVar[np.ndarray] = np.array([0.5, 0.5])


class ThreePtQuadrature(Quadrature):
    """Quadrature with three points per triangle, so six per pixel."""

    quad_pt_coords: ClassVar[np.ndarray] = np.array(
        [[4 / 6, 1 / 6], [1 / 6, 1 / 6], [1 / 6, 4 / 6], [2 / 6, 5 / 6], [5 / 6, 5 / 6], [5 / 6, 2 / 6]]
    )
    quad_pt_weights: ClassVar[np.ndarray] = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
