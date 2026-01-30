"""
Self-affine rough surface generation.
"""

import dataclasses as dc

import numpy as np
import numpy.linalg as la
import numpy.fft as fft
import numpy.random as random

from a_package.domain import Field, field_component_ax


@dc.dataclass(init=True)
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

    def mapto_isotropic_psd(self, q: Field):
        """
        Get the isotropic power spectral density (psd) of a given wavenumber.

        Parameters
        ----------
        q : Field
            Wavenumber in radians, i.e. 2*pi / wavelength.
        """
        # isotropic, only the magnitude matters
        wavenumber = la.norm(q, ord=2, axis=field_component_ax, keepdims=True)

        # Find three regimes
        constant = wavenumber < self.qR
        self_affine = (wavenumber >= self.qR) & (wavenumber < self.qS)
        omitted = wavenumber >= self.qS

        # Evaluate accordingly
        psd = np.full_like(wavenumber, np.nan)
        psd[constant] = self.C0 * self.qR ** (-2 - 2 * self.H)
        psd[self_affine] = self.C0 * wavenumber[self_affine] ** (-2 - 2 * self.H)
        psd[omitted] = 0

        # Return both in convenience of plotting
        return wavenumber, psd


def psd_to_height(psd: Field, rng=None, seed=None):
    """Convert power spectral density to height field via inverse FFT."""
    # <h^2> corresponding to <PSD>, thus, take the square-root to match overall amplitude
    h_amp = np.sqrt(psd)

    # impose some random phase angle following uniform distribution
    if rng is None:
        rng = random.default_rng(seed)
    phase_angle = np.exp(1j * rng.uniform(0, 2 * np.pi, psd.shape))

    # only the sinusoidal is needed
    return fft.ifft2(h_amp * phase_angle).real
