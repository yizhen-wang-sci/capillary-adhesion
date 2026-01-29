import typing
import functools
import operator
import dataclasses as dc

import numpy as np
import numpy.fft as fft


@dc.dataclass
class Grid:
    """A discrete space in 2D."""

    lengths: typing.Sequence[float]
    nb_elements: typing.Sequence[int]

    def __post_init__(self):
        self.element_sizes = [l / n for [l, n] in zip(self.lengths, self.nb_elements)]
        self.element_area = functools.reduce(operator.mul, self.element_sizes, 1.)

    @property
    def is_in_parallel(self):
        return False

    def form_index_axis(self, ax_index: int):
        return np.arange(self.nb_elements[ax_index])

    def form_index_mesh(self):
        return np.meshgrid(self.form_index_axis(0), self.form_index_axis(1))

    def form_nodal_axis(self, ax_index: int, with_endpoint: bool = False):
        d = self.element_sizes[ax_index]
        n = self.nb_elements[ax_index]
        if with_endpoint:
            n += 1
        return np.arange(n) * d

    def form_nodal_mesh(self, with_endpoint: bool = False):
        return np.meshgrid(self.form_nodal_axis(0, with_endpoint), self.form_nodal_axis(1, with_endpoint))

    def form_spectral_axis(self, ax_index: int):
        d = self.element_sizes[ax_index]
        n = self.nb_elements[ax_index]
        return (2 * np.pi) * fft.fftfreq(n, d)

    def form_spectral_mesh(self):
        return np.meshgrid(self.form_spectral_axis(0), self.form_spectral_axis(1))
