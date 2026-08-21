"""The discrete space, and the coordinate systems over it."""

from typing import Sequence

import numpy as np
import numpy.fft as fft
import muGrid
from NuMPI import MPI


class Grid:
    """A 2D regular grid, the coordinate foundation for fields."""

    def __init__(self, nb_grid_pts: Sequence[int], lengths: Sequence[float] | None = None,
                 decomposition: muGrid.CartesianDecomposition | None= None):
        """Set up the grid, deriving the element sizes from the domain lengths.

        Args:
            nb_grid_pts: Number of grid points along each dimension. Its length sets the number
                of spatial dimensions.
            lengths: Physical length of the domain along each dimension, 1.0 in each by default.
            decomposition: How the domain is split across processes. Defaults to no split, where
                every process holds a grid spanning the whole global domain.

        Raises:
            ValueError: If `lengths` and `nb_grid_pts` have different dimensions.
        """
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
        """Decompose a grid, such that each process gets a subdomain of the same global domain.

        Args:
            nb_subdomains: Number of subdomains along each dimension.
            nb_ghost_layers: Number of ghost layers along each dimension, applied at both ends,
                0 in each by default.
            communicator: Communicator across whose ranks the subdomains are spread,
                `MPI.COMM_SELF` by default. A `muGrid.Communicator` is accepted as well.

        Returns:
            The new decomposition, which also replaces the grid's own.

        Raises:
            ValueError: If `nb_subdomains` or `nb_ghost_layers` has a different number of
                dimensions than the grid, or if the communicator holds fewer processes than
                `nb_subdomains` demands.
        """
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

    def get_local(self, field):
        """Return the local part of a field.

        Args:
            field: A field spanning the whole global domain, with the spatial axes last.

        Returns:
            The part of `field` belonging to this rank's subdomain.
        """
        return field[(..., *self.decomposition.icoords)]

    # FIXME: now there shall be a difference between local and global indices
    # where the global indices are from decomposition.subdomain_locations and do not exceed the nb_domain_grid_pts.
    # While the local ones are simply from 0 to decomposition.nb_subdomain_grid_pts (endpoint).

    # =========================================================================
    # Index: 0, 1, 2, ..., N-1
    # =========================================================================

    def form_index_axis(self, ax_index: int, endpoint: bool = False):
        """Indices along the specified axis: 0, 1, 2, ..., N-1.

        Args:
            ax_index: Which dimension.
            endpoint: Whether to append one index past the last.

        Returns:
            The indices along that dimension.
        """
        axis = np.arange(self.nb_domain_grid_pts[ax_index])
        if endpoint:
            axis = np.append(axis, self.nb_domain_grid_pts[ax_index])
        return axis

    def form_index_mesh(self, endpoint: bool = False):
        """Index coordinates over the whole grid.

        Args:
            endpoint: Whether to append one index past the last.

        Returns:
            One index mesh per dimension, in "ij" order.
        """
        return np.meshgrid(self.form_index_axis(0, endpoint), self.form_index_axis(1, endpoint),
                           indexing="ij")

    # =========================================================================
    # Spatial: 0, d, 2d, ..., (N-1)d
    # =========================================================================

    def form_spatial_axis(self, ax_index: int, endpoint: bool = False):
        """Spatial coordinates along the specified axis: 0, d, 2d, ..., (N-1)d.

        Args:
            ax_index: Which dimension.
            endpoint: Whether to append one point past the last.

        Returns:
            The coordinates along that dimension, spaced by its element size.
        """
        d = self.element_sizes[ax_index]
        n = self.nb_domain_grid_pts[ax_index]
        if endpoint:
            n += 1
        return np.arange(n) * d

    def form_spatial_mesh(self, endpoint: bool = False):
        """Spatial coordinates over the whole grid.

        Args:
            endpoint: Whether to append one point past the last.

        Returns:
            One coordinate mesh per dimension, in "ij" order.
        """
        return np.meshgrid(self.form_spatial_axis(0, endpoint), self.form_spatial_axis(1, endpoint),
                           indexing="ij")

    # =========================================================================
    # Spectral: 2π / (N * pixel_size * ref_scale) * fftfreq indices
    # =========================================================================

    def form_spectral_axis(self, ax_index: int):
        """Spectral wavenumbers along the specified axis, in FFT order.

        Args:
            ax_index: Which dimension.

        Returns:
            Angular wavenumbers, 2*pi times the FFT frequencies.
        """
        n = self.nb_domain_grid_pts[ax_index]
        d = self.element_sizes[ax_index]
        return (2 * np.pi) * fft.fftfreq(n, d)

    def form_spectral_mesh(self):
        """Spectral coordinates over the whole grid, in FFT order.

        Returns:
            One wavenumber mesh per dimension, in "ij" order.
        """
        return np.meshgrid(self.form_spectral_axis(0), self.form_spectral_axis(1), indexing="ij")


def factorize_closest(value: int, nb_factor: int):
    """The maximal combination of `nb_factor` integers whose product does not exceed `value`.

    Args:
        value: The value to factorize.
        nb_factor: How many factors to split it into.

    Returns:
        The factors.

    Raises:
        ValueError: If no such combination exists.
    """
    factors = []
    for root_degree in range(nb_factor, 0, -1):
        max_divisor = int(value ** (1 / root_degree))
        factors.append(max_divisor)
        value //= max_divisor
    # FIXME: muGrid can't handle empty subdomain yet.
    if np.multiply.reduce(factors) < value:
        raise ValueError("Cannot factorize value into nb_factor integers without empty subdomains.")
    return factors
