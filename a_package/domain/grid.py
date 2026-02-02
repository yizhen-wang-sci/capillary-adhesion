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

    # =========================================================================
    # Index: 0, 1, 2, ..., N-1
    # =========================================================================

    def form_index_axis(self, ax_index: int):
        """Return indices: 0, 1, 2, ..., N-1."""
        return np.arange(self.nb_elements[ax_index])

    def form_index_mesh(self):
        return np.meshgrid(self.form_index_axis(0), self.form_index_axis(1))

    # =========================================================================
    # Spatial: 0, d, 2d, ..., (N-1)d
    # =========================================================================

    def form_spatial_axis(self, ax_index: int, with_endpoint: bool = False):
        """Return spatial coordinates: 0, d, 2d, ..., (N-1)d."""
        d = self.element_sizes[ax_index]
        n = self.nb_elements[ax_index]
        if with_endpoint:
            n += 1
        return np.arange(n) * d

    def form_spatial_mesh(self, with_endpoint: bool = False):
        return np.meshgrid(self.form_spatial_axis(0, with_endpoint), self.form_spatial_axis(1, with_endpoint))

    # =========================================================================
    # Spectral: 2π / (N * pixel_size * ref_scale) * fftfreq indices
    # =========================================================================

    def form_spectral_axis(self, ax_index: int):
        """Return spectral wavenumbers in FFT order."""
        n = self.nb_elements[ax_index]
        d = self.element_sizes[ax_index]
        return (2 * np.pi) * fft.fftfreq(n, d)

    def form_spectral_mesh(self):
        return np.meshgrid(self.form_spectral_axis(0), self.form_spectral_axis(1))

    # =========================================================================
    # Rescale by element_area
    # =========================================================================

    def rescale(self, array: np.ndarray, direction: str):
        """
        Rescale array by element_area.

        direction='forward': multiply by element_area (e.g., integral, before fft2)
        direction='backward': divide by element_area (e.g., after ifft2)
        """
        if direction == 'forward':
            return array * self.element_area
        elif direction == 'backward':
            return array / self.element_area
        else:
            raise ValueError(f"direction must be 'forward' or 'backward', got {direction}")
