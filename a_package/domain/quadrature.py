import logging
from abc import ABC
from typing import ClassVar

import numpy as np
from NuMPI import MPI


from .field import Field, field_element_axs


logger = logging.getLogger(__name__)


class Quadrature(ABC):
    """Quadrature for numerical approximating an integral.

    The implementation assumes a regular grid.

    Every concrete subclass must define both `quad_pt_coords` and
    `quad_pt_weights` as class attributes. They are validated and
    converted to read-only NumPy arrays once, at class-definition
    time.
    """

    quad_pt_coords:  ClassVar[np.ndarray]
    quad_pt_weights: ClassVar[np.ndarray]
    _REQUIRED = ("quad_pt_coords", "quad_pt_weights")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Ensure the subclass defines all required attributes
        missing = [name for name in cls._REQUIRED if name not in cls.__dict__]
        if missing:
            raise TypeError(f"{cls.__name__} must define class attribute(s): {', '.join(missing)}")

        # Value validation
        coords  = np.asarray(cls.quad_pt_coords,  dtype=float)
        weights = np.asarray(cls.quad_pt_weights, dtype=float)
        if coords.ndim != 2:
            raise ValueError(f"{cls.__name__}: quad_pt_coords must be 2-D, got shape {coords.shape}")
        if weights.ndim != 1:
            raise ValueError(f"{cls.__name__}: quad_pt_weights must be 1-D, got shape {weights.shape}")
        if coords.shape[0] != weights.size:
            raise ValueError(f"{cls.__name__}: number of quadrature points ({coords.shape[0]}) must "
                             f"match number of weights ({weights.size})")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"{cls.__name__}: quadrature weights must sum to 1, got {weights.sum()}")

        # Freeze and store the validated arrays back to the class
        coords.flags.writeable = False
        weights.flags.writeable = False
        cls.quad_pt_coords = coords
        cls.quad_pt_weights = weights

    def __init__(self, communicator=MPI.COMM_SELF):
        if type(self) is Quadrature:
            raise TypeError("Quadrature is abstract, instantiate a subclass instead.")
        self._communicator = communicator

    @property
    def nb_quad_pts(self):
        return self.quad_pt_weights.size

    def integrate(self, field: Field, element_area: float = 1.0):
        # Regular grid -> element area factors out
        element_sum = element_area * np.sum(field, axis=field_element_axs)
        local = np.einsum("s, cs-> c", self.quad_pt_weights, element_sum)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"rank={self._communicator.rank}, local integral={local}")
        return self._communicator.allreduce(local, op=MPI.SUM)

    def propag_integral_weight(self, field: Field, element_area: float = 1.0):
        return element_area * np.einsum("s, cs...-> cs...", self.quad_pt_weights, field)


class NodalQuadrature(Quadrature):
    """Quadrature by summing up nodal values."""
    quad_pt_coords = [[0., 0.]]
    quad_pt_weights = [1.]


class CentroidQuadrature(Quadrature):
    """Numerical quadrature with points located at the centroid of the two triangular elements of each pixel."""
    quad_pt_coords = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]
    quad_pt_weights = [0.5, 0.5]


class ThreePtQuadrature(Quadrature):
    quad_pt_coords = [[4 / 6, 1 / 6], [1 / 6, 1 / 6], [1 / 6, 4 / 6], [2 / 6, 5 / 6], [5 / 6, 5 / 6], [5 / 6, 2 / 6]]
    quad_pt_weights = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
