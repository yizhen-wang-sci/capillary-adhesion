import typing
import functools
import operator

import numpy as np
import numpy.fft as fft
import muGrid

try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
except ModuleNotFoundError:
    _comm = None
communicator = muGrid.Communicator(_comm)


class Grid:
    """A discrete space in 2D."""

    lengths: tuple[float, ...]
    nb_elements: tuple[int, ...]
    subdomain_base: tuple[int, ...]

    def __init__(
            self, lengths: typing.Sequence[float],
            nb_elements: typing.Sequence[int],
            nb_subdomains: typing.Sequence[int] | None = None,
            nb_ghost_layers: typing.Sequence[int] | None = None) -> None:
        self.lengths_global = lengths
        self.nb_elements_global = nb_elements
        self.element_sizes = [l / n for [l, n] in zip(lengths, nb_elements)]
        self.element_area = functools.reduce(operator.mul, self.element_sizes, 1.)

        if nb_subdomains is None:
            nb_subdomains = factorize_closest(communicator.size, self.nb_spatial_dims)
        # ghost layers, set all to 1 by default
        if nb_ghost_layers is None:
            nb_ghost_layers = [1] * self.nb_spatial_dims
        self.nb_ghost_layers = nb_ghost_layers

        self.decomposition = muGrid.CartesianDecomposition(
            communicator, nb_elements, nb_subdomains, nb_ghost_layers, nb_ghost_layers)
        self.lengths = tuple(d * n for [d, n] in zip(self.element_sizes, self.nb_elements))
        self.sync_field: typing.Callable[["muGrid.Field"]] = self.decomposition.communicate_ghosts

    @property
    def nb_spatial_dims(self):
        return len(self.lengths_global)

    @property
    def nb_elements(self):
        return tuple(self.decomposition.nb_subdomain_grid_pts)

    @property
    def subdomain_base(self):
        return tuple(self.decomposition.subdomain_locations)

    @property
    def subdomain_slice(self):
        return (Ellipsis, *(slice(b, b+n) for [b, n] in zip(self.subdomain_base, self.nb_elements)))

    def form_index_axis(self, ax_index: int):
        return self.subdomain_base[ax_index] + np.arange(self.nb_elements[ax_index])

    def form_index_mesh(self):
        return self.decomposition.icoords

    def form_nodal_axis(self, ax_index: int):
        base = self.subdomain_base[ax_index] / self.nb_elements_global[ax_index] * self.lengths_global[ax_index]
        return base + np.arange(self.nb_elements[ax_index]) * self.element_sizes[ax_index]

    def form_nodal_mesh(self):
        return self.decomposition.coords * np.asarray(self.lengths_global)[..., np.newaxis, np.newaxis]

    # FIXME: muFFT?
    def form_spectral_axis(self, ax_index: int):
        d = self.element_sizes[ax_index]
        n = self.nb_elements[ax_index]
        return (2 * np.pi) * fft.fftfreq(n, d)

    # FIXME: muFFT?
    def form_spectral_mesh(self):
        return np.meshgrid(self.form_spectral_axis(0), self.form_spectral_axis(1))


def factorize_closest(value: int, nb_ints: int):
    """Find the maximal combination of nb_ints integers whose product is less or equal to value."""
    nb_divisions = []
    for root_degree in range(nb_ints, 0, -1):
        max_divisor = int(value ** (1 / root_degree))
        nb_divisions.append(max_divisor)
        value //= max_divisor
    return nb_divisions
