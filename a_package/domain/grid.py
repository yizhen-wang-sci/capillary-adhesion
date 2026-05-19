import functools
import operator
from typing import Sequence

import numpy as np
import numpy.fft as fft
import muGrid


class Grid:
    """A discrete space in 2D.

    A thin wrapper of muGrid instances.
    Since muGrid doesn't know the domain lengths, we provide them here.
    """

    def __init__(self, nb_grid_pts: Sequence[int], lengths: Sequence[float] | None = None):
        if lengths is None:
            lengths = [1.0] * len(nb_grid_pts)
        if len(lengths) != len(nb_grid_pts):
            raise ValueError("lengths and nb_grid_pts must have compatible dimensions.")
        self.domain_lengths = tuple(lengths)
        self.nb_domain_grid_pts = tuple(nb_grid_pts)

        self.nb_spatial_dim = len(self.nb_domain_grid_pts)
        self.element_sizes = [l / n for [l, n] in zip(self.domain_lengths, self.nb_domain_grid_pts)]
        self.element_area = functools.reduce(operator.mul, self.element_sizes, 1.)

    def decompose(self,
                  nb_subdomains: Sequence[int] | None = None,
                  nb_ghost_layers: Sequence[int] | None = None,
                  communicator=None):

        # default value of nb_subdomains, set to all 1 so it is equivalent to no decomposition
        if nb_subdomains is None:
            nb_subdomains = [1] * self.nb_spatial_dim

        # default value of nb_ghost_layers, set to all 0 so it is equivalent to no decomposition
        if nb_ghost_layers is None:
            nb_ghost_layers = [0] * self.nb_spatial_dim

        # Wrap the communicator in a muGrid.Communicator object. The constructor has a mechanism
        # to avoid overhead if the communicator is already a muGrid.Communicator object.
        communicator = muGrid.Communicator(communicator)

        return muGrid.CartesianDecomposition(
            communicator, self.nb_domain_grid_pts, tuple(nb_subdomains), tuple(nb_ghost_layers), tuple(nb_ghost_layers))

    # =========================================================================
    # Index: 0, 1, 2, ..., N-1
    # =========================================================================

    def form_index_axis(self, ax_index: int, endpoint: bool = False):
        """Return indices: 0, 1, 2, ..., N-1."""
        axis = np.arange(self.nb_domain_grid_pts[ax_index])
        if endpoint:
            axis = np.append(axis, self.nb_domain_grid_pts[ax_index])
        return axis

    def form_index_mesh(self, endpoint: bool = False):
        return np.meshgrid(self.form_index_axis(0, endpoint), self.form_index_axis(1, endpoint))

    # =========================================================================
    # Spatial: 0, d, 2d, ..., (N-1)d
    # =========================================================================

    def form_spatial_axis(self, ax_index: int, endpoint: bool = False):
        """Return spatial coordinates: 0, d, 2d, ..., (N-1)d."""
        d = self.element_sizes[ax_index]
        n = self.nb_domain_grid_pts[ax_index]
        if endpoint:
            n += 1
        return np.arange(n) * d

    def form_spatial_mesh(self, endpoint: bool = False):
        return np.meshgrid(self.form_spatial_axis(0, endpoint), self.form_spatial_axis(1, endpoint))

    # =========================================================================
    # Spectral: 2π / (N * pixel_size * ref_scale) * fftfreq indices
    # =========================================================================

    def form_spectral_axis(self, ax_index: int):
        """Return spectral wavenumbers in FFT order."""
        n = self.nb_domain_grid_pts[ax_index]
        d = self.element_sizes[ax_index]
        return (2 * np.pi) * fft.fftfreq(n, d)

    def form_spectral_mesh(self):
        return np.meshgrid(self.form_spectral_axis(0), self.form_spectral_axis(1))


def factorize_closest(value: int, nb_factor: int):
    """Find the maximal combination of nb_factor integers whose product is less or equal to value."""
    factors = []
    for root_degree in range(nb_factor, 0, -1):
        max_divisor = int(value ** (1 / root_degree))
        factors.append(max_divisor)
        value //= max_divisor
    return factors
