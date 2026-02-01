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
    qR: float
    """Roll-off (angular) wavenumber"""
    qS: float
    """Cut-off (angular) wavenumber"""
    H: float
    """Hurst exponent"""

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
        constant = wavenumber < self.qR
        self_affine = (wavenumber >= self.qR) & (wavenumber < self.qS)
        omitted = wavenumber >= self.qS

        # Evaluate accordingly
        psd = np.full_like(wavenumber, np.nan, dtype=float)
        psd[constant] = self.C0 * self.qR ** (-2 - 2 * self.H)
        psd[self_affine] = self.C0 * wavenumber[self_affine] ** (-2 - 2 * self.H)
        psd[omitted] = 0

        return psd


def psd_to_height(psd: np.ndarray, seed: int | None = None, spatial_axes: Sequence[int] | None = None):
    """Convert power spectral density to height field via inverse FFT.

    Pass seed for reproducibility; None uses random seed.
    """
    # <h^2> corresponding to <PSD>, thus, take the square-root to match overall amplitude
    amplitude = np.sqrt(psd)

    # impose some random phase angle following uniform distribution
    rng = random.default_rng(seed)
    phase_angle = np.exp(1j * rng.uniform(0, 2 * np.pi, psd.shape))

    # transform back to real space
    return fft.ifft2(amplitude * phase_angle, axes=spatial_axes).real
