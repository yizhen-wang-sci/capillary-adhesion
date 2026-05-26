from typing import Sequence

import numpy as np
import numpy.fft as fft
import muGrid
from NuMPI import MPI


class Grid:
    """A discrete space in 2D.

    A thin wrapper of muGrid instances.
    Since muGrid doesn't know the domain lengths, we provide them here.
    """

    def __init__(self, nb_grid_pts: Sequence[int], lengths: Sequence[float] | None = None,
                 decomposition: muGrid.CartesianDecomposition | None= None):
        self.nb_domain_grid_pts = tuple(nb_grid_pts)
        self.nb_spatial_dim = len(self.nb_domain_grid_pts)

        if lengths is None:
            # default to 1.0 in each dimension
            lengths = (1.0,) * len(nb_grid_pts)
        if len(lengths) != len(nb_grid_pts):
            raise ValueError("lengths and nb_grid_pts must have compatible dimensions.")
        self.domain_lengths = tuple(lengths)

        self.element_sizes = [l / n for l, n in zip(self.domain_lengths, self.nb_domain_grid_pts)]
        self.element_area = np.multiply.reduce(self.element_sizes, initial=1.)

        if decomposition is None:
            # default to no decomposition, where all processes have its grid representing the same global domain.
            decomposition = muGrid.CartesianDecomposition(muGrid.Communicator(MPI.COMM_SELF),
                                                          list(self.nb_domain_grid_pts), [1] * self.nb_spatial_dim,
                                                          [0] * self.nb_spatial_dim, [0] * self.nb_spatial_dim)
        self.decomposition = decomposition

    def decompose(self, nb_subdomains: Sequence[int],
                  nb_ghost_layers: Sequence[int] | None = None, communicator = MPI.COMM_SELF):
        """Decompose a grid, such that each process gets a subdomain of the same global domain."""
        if len(nb_subdomains) != self.nb_spatial_dim:
            raise ValueError(f"nb_subdomains must have the same dimension as nb_grid_pts, got {len(nb_subdomains)} "
                             f"and {self.nb_spatial_dim}")

        if nb_ghost_layers is None:
            # default to all 0 in each dimension
            nb_ghost_layers = [0] * self.nb_spatial_dim
        if len(nb_ghost_layers) != self.nb_spatial_dim:
            raise ValueError(f"nb_ghost_layers must have the same dimension as nb_grid_pts, got {len(nb_ghost_layers)} "
                             f"and {self.nb_spatial_dim}")

        if communicator.Get_size() < np.multiply.reduce(nb_subdomains):
            raise ValueError(f"The number of processes ({communicator.Get_size()}) is less than is demanded by "
                             f"nb_subdomains ({'x'.join(str(n) for n in nb_subdomains)}).")
        # Wrap the communicator in a muGrid.Communicator object. The constructor has a mechanism
        # to avoid overhead if the communicator is already a muGrid.Communicator object.
        communicator = muGrid.Communicator(communicator)

        self.decomposition = muGrid.CartesianDecomposition(communicator, list(self.nb_domain_grid_pts),
                                                           list(nb_subdomains), list(nb_ghost_layers),
                                                           list(nb_ghost_layers))
        return self.decomposition

    # FIXME: now there shall be a difference between local and global indices
    # where the global indices are from decomposition.subdomain_locations and do not exceed the nb_domain_grid_pts.
    # While the local ones are simply from 0 to decomposition.nb_subdomain_grid_pts (endpoint).

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
        return np.meshgrid(self.form_index_axis(0, endpoint), self.form_index_axis(1, endpoint),
                           indexing="ij")

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
        return np.meshgrid(self.form_spatial_axis(0, endpoint), self.form_spatial_axis(1, endpoint),
                           indexing="ij")

    # =========================================================================
    # Spectral: 2π / (N * pixel_size * ref_scale) * fftfreq indices
    # =========================================================================

    def form_spectral_axis(self, ax_index: int):
        """Return spectral wavenumbers in FFT order."""
        n = self.nb_domain_grid_pts[ax_index]
        d = self.element_sizes[ax_index]
        return (2 * np.pi) * fft.fftfreq(n, d)

    def form_spectral_mesh(self):
        return np.meshgrid(self.form_spectral_axis(0), self.form_spectral_axis(1), indexing="ij")


def factorize_closest(value: int, nb_factor: int):
    """Find the maximal combination of nb_factor integers whose product is less or equal to value."""
    factors = []
    for root_degree in range(nb_factor, 0, -1):
        max_divisor = int(value ** (1 / root_degree))
        factors.append(max_divisor)
        value //= max_divisor
    return factors
