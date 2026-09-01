"""The discrete space, and the coordinate systems over it."""

from collections.abc import Sequence

import muGrid
import numpy as np
from NuMPI import MPI
from numpy import fft


class Grid:
    """A 2D regular grid, the coordinate foundation for fields."""

    def __init__(
        self,
        nb_grid_pts: Sequence[int],
        lengths: Sequence[float] | None = None,
        decomposition = None,
    ):
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
        self.element_area = np.multiply.reduce(self.element_sizes, initial=1.0)

        if decomposition is None:
            # default to no decomposition, where all processes have its grid representing
            # the same global domain.
            decomposition = muGrid.CartesianDecomposition(
                muGrid.Communicator(MPI.COMM_SELF),
                list(self.nb_domain_grid_pts),
                [1] * self.nb_spatial_dim,
                [0] * self.nb_spatial_dim,
                [0] * self.nb_spatial_dim,
            )
        self._decomposition = decomposition
        self._supports_spectrum = isinstance(decomposition, muGrid.FFTEngine)

    def decompose(
        self,
        nb_subdomains: Sequence[int] | None = None,
        nb_ghost_layers: Sequence[int] | None = None,
        communicator=MPI.COMM_SELF,
    ):
        """Decompose a grid, such that each process gets a subdomain of the same global domain.

        Args:
            nb_subdomains: Number of subdomains along each dimension, or along the last alone,
                the leading ones being left whole. Defaults to one subdomain per process along
                the last dimension. A split along the last dimension alone is carried out by a
                spectral transform, which `convolve` and `form_local_spectral_mesh` then work over.
            nb_ghost_layers: Number of ghost layers along each dimension, applied at both ends,
                0 in each by default.
            communicator: Communicator across whose ranks the subdomains are spread,
                `MPI.COMM_SELF` by default. A `muGrid.Communicator` is accepted as well.

        Returns:
            The new decomposition, which also replaces the grid's own.

        Raises:
            ValueError: If `nb_subdomains` has more dimensions than the grid, if it divides any
                dimension into a non-positive number of subdomains or into more subdomains than
                that dimension has grid points, if the subdomains do not number the processes,
                or if `nb_ghost_layers` has a different number of dimensions than the grid.
        """
        # FIXME: it doesn't consider thoroughly in case of a 3D grid.
        # Default to decomposing solely the last dimension
        if nb_subdomains is None:
            nb_subdomains = [1, communicator.size]
        # Check that the number of subdomains has the correct spatial dimensions
        if len(nb_subdomains) != self.nb_spatial_dim:
            raise ValueError(
                f"The number of subdomains ({'x'.join(str(nb) for nb in nb_subdomains)}) exceeds "
                f"the number of spatial dimensions ({self.nb_spatial_dim})."
            )
        # Check that the number of subdomains is positive in every dimension
        if any(nb <= 0 for nb in nb_subdomains):
            raise ValueError(
                f"The number of subdomains ({'x'.join(str(nb) for nb in nb_subdomains)}) is not positive "
                f"in every dimension."
            )
        # Check that no subdomain is left empty
        if any(nb_s > nb_g for nb_s, nb_g in zip(nb_subdomains, self.nb_domain_grid_pts)):
            raise ValueError(
                f"The number of subdomains ({'x'.join(str(nb) for nb in nb_subdomains)}) exceeds the "
                f"number of grid points ({'x'.join(str(nb) for nb in self.nb_domain_grid_pts)})."
            )
        if np.multiply.reduce(nb_subdomains) != communicator.size:
            raise ValueError(
                f"The number of subdomains ({'x'.join(str(nb) for nb in nb_subdomains)}) is not the "
                f"number of processes ({communicator.size})."
            )

        # Default to no ghost layer in all dimensions
        if nb_ghost_layers is None:
            nb_ghost_layers = [0] * self.nb_spatial_dim
        if len(nb_ghost_layers) != self.nb_spatial_dim:
            raise ValueError(
                f"nb_ghost_layers must have the same dimension as nb_grid_pts, got {len(nb_ghost_layers)} "
                f"and {self.nb_spatial_dim}"
            )

        # Wrap the communicator in a muGrid.Communicator object. The constructor has a mechanism
        # to avoid overhead if the communicator is already a muGrid.Communicator object.
        communicator = muGrid.Communicator(communicator)

        # If nb_subdomains satisfies the requirement of FFTEngine, use it as backend
        try:
            self._decomposition = muGrid.FFTEngine(
                list(self.nb_domain_grid_pts),
                communicator,
                list(nb_ghost_layers),
                list(nb_ghost_layers),
            )
            if self._decomposition.nb_subdivisions != nb_subdomains:
                raise RuntimeWarning()
        # Otherwise, use CartesianDecomposition as backend
        except RuntimeWarning:
            self._decomposition = muGrid.CartesianDecomposition(
                communicator,
                list(self.nb_domain_grid_pts),
                list(nb_subdomains),
                list(nb_ghost_layers),
                list(nb_ghost_layers),
            )
        self._supports_spectrum = isinstance(self._decomposition, muGrid.FFTEngine)
        return self._decomposition

    @property
    def decomposition(self):
        """How the domain is split across processes."""
        return self._decomposition

    def owned_layout(self):
        """How this rank's part of the domain sits inside it, ghost layers excluded.

        Returns:
            The shape of the whole domain, the shape of the part this rank is the authority
            for, and where that part begins in the index space of the domain, keyed by name.
        """
        return {
            "domain_shape": tuple(self._decomposition.nb_domain_grid_pts),
            "owned_shape": tuple(self._decomposition.nb_subdomain_grid_pts),
            "owned_offset": tuple(self._decomposition.subdomain_locations),
        }

    @property
    def collection_real(self):
        """The real field collection over this rank's subdomain."""
        if self._supports_spectrum:
            return self._decomposition.real_space_collection
        return self._decomposition.collection

    # Keep old name until all scripts are updated
    collection = collection_real

    @property
    def collection_spectral(self):
        """The spectral field collection over this rank's share of the spectrum.

        Raises:
            AttributeError: If the decomposition does not support spectral decomposition.
        """
        if self._supports_spectrum:
            return self._decomposition.fourier_space_collection
        raise AttributeError("The grid must not decompose the first dimension to support spectral decomposition.")

    def get_local(self, field: np.ndarray):
        """Return the local part of a field.

        Args:
            field: A field spanning the whole global domain, with the spatial axes last.

        Returns:
            The part of `field` belonging to this rank's subdomain.
        """
        return field[(..., *self.decomposition.icoords)]

    def form_local_spectral_mesh(self):
        """Spectral coordinates over this rank's share of the spectrum a transform spans.

        Returns:
            One wavenumber mesh per dimension, stacked along the leading axis.

        Raises:
            AttributeError: If the decomposition does not support spectral decomposition.
        """
        if not self._supports_spectrum:
            raise AttributeError("The grid must not decompose the first dimension to support spectral decomposition.")
        cycles_per_point = np.asarray(self._decomposition.fftfreq)
        return np.stack(
            [
                (2 * np.pi) * axis * nb_pts / length
                for axis, nb_pts, length in zip(cycles_per_point, self.nb_domain_grid_pts, self.domain_lengths)
            ]
        )

    def convolve(self, field: np.ndarray, kernel: np.ndarray):
        """Convolve a field over the whole domain with a kernel given in spectral space.

        Args:
            field: Values over this rank's subdomain, with the element axes last.
            kernel: The kernel in spectral space, over the mesh `form_local_spectral_mesh` spans.

        Returns:
            The convolution, over this rank's subdomain, shaped like `field`.

        Raises:
            AttributeError: If the decomposition does not support spectral decomposition.
        """
        if not self._supports_spectrum:
            raise AttributeError("The grid must not decompose the first dimension to support spectral decomposition.")

        real = self._decomposition.real_space_field("convolution_real", 1)
        spectral = self._decomposition.fourier_space_field("convolution_spectral", 1)
        real.s[...] = np.reshape(field, real.s.shape)

        self._decomposition.fft(real, spectral)
        spectral.s[...] *= kernel
        self._decomposition.ifft(spectral, real)
        real.s[...] *= self._decomposition.normalisation

        return np.reshape(np.array(real.s), np.shape(field))

    # FIXME: now there shall be a difference between local and global indices
    # where the global indices are from decomposition.subdomain_locations and do not exceed
    # the nb_domain_grid_pts.
    # While the local ones are simply from 0 to decomposition.nb_subdomain_grid_pts (endpoint).

    # =========================================================================
    # Index: 0, 1, 2, ..., N-1

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
        return np.meshgrid(self.form_index_axis(0, endpoint), self.form_index_axis(1, endpoint), indexing="ij")

    # =========================================================================
    # Spatial: 0, d, 2d, ..., (N-1)d

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
        return np.meshgrid(self.form_spatial_axis(0, endpoint), self.form_spatial_axis(1, endpoint), indexing="ij")

    # =========================================================================
    # Spectral: 2π / (N * pixel_size * ref_scale) * fftfreq indices

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
        raise ValueError(f"Cannot factorize {value} into {nb_factor} integers.")
    return factors
