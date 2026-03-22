"""
Self-affine rough surface generation.
"""

import dataclasses as dc
from typing import Sequence

import numpy as np
import numpy.linalg as linalg
import numpy.fft as fft
import numpy.random as random


@dc.dataclass(init=True, frozen=True)
class SelfAffineRoughness:
    """Parameters defining self-affine roughness spectrum."""
    C0: float
    """Prefactor"""
    H: float
    """Hurst exponent"""
    qR: float
    """The (angular) wavenumber below which the PSD keeps constant, above which the PSD rolls off."""
    qS: float
    """The (angular) wavenumber above which the PSD is negligible."""
    qT: float = 2*np.pi
    """The (angular) wavenumber below which the PSD is terminated. Defaults to 2π (1 cycle over unit length)."""

    def mapto_isotropic_psd(self, wavevector: np.ndarray, component_axis: int | None = None):
        """
        Get the isotropic power spectral density (psd) of a given wavenumber.

        Parameters
        ----------
        wavevector : NumPy
            Wavevector with components in radians, i.e. 2*pi / wavelength.
        component_axis : int | None
            If None, wavevector is treated as single component wavenumber.
            If int, compute magnitude via norm along this axis.
        """
        if component_axis is None:
            wavenumber = wavevector
        else:
            wavenumber = linalg.norm(wavevector, ord=2, axis=component_axis)

        # Find three regimes
        constant = (wavenumber >= self.qT) & (wavenumber < self.qR)
        self_affine = (wavenumber >= self.qR) & (wavenumber < self.qS)
        zeroed = (wavenumber < self.qT) | (wavenumber >= self.qS)

        # Evaluate accordingly
        psd = np.full_like(wavenumber, np.nan, dtype=float)
        psd[constant] = self.C0 * self.qR ** (-2 - 2 * self.H)
        psd[self_affine] = self.C0 * wavenumber[self_affine] ** (-2 - 2 * self.H)
        psd[zeroed] = 0

        # Ensure mean value is zero
        psd[wavenumber == 0] = 0

        return psd


def psd_to_height(psd: np.ndarray, seed: int | None = None, spatial_axes: Sequence[int] | None = None):
    """Convert power spectral density to height field via inverse FFT.

    seed: seed passed to RNG for reproducibility; None uses random seed.
    spatial_axes: axes along which to apply FFT; None (by NumPy) uses the last 2 axes as spatial.
    """
    # <h^2> corresponding to <PSD>, thus, take the square-root to match overall amplitude
    amplitude = np.sqrt(psd)

    # impose some random phase angle following uniform distribution
    rng = random.default_rng(seed)
    phase_angle = np.exp(1j * rng.uniform(0, 2 * np.pi, psd.shape))

    # transform back to real space
    return fft.ifft2(amplitude * phase_angle, axes=spatial_axes).real

    # cancels out NumPy's prefactor of N1*N2
    # FIXME: when spatial_axes=None
    # nb_grid_pts = np.multiply.reduce(np.take(psd.shape, spatial_axes))
