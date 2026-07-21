"""
Self-affine rough surface generation.
"""

import dataclasses as dc
from typing import Sequence

import numpy as np
import numpy.linalg as linalg
import numpy.fft as fft
import numpy.random as random

from a_package.domain import Grid


@dc.dataclass(init=True)
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

    def __post_init__(self):
        if not (0 < self.qT <= self.qR <= self.qS):
            raise ValueError("The three wavenumbers must be positive and ordered as qT <= qR <= qS.")

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

        return psd

    def correct_prefactor_by_rms_height(self, value: float):
        self.C0 = 4 * np.pi * self.H * value ** 2 / ((1 + self.H) * self.qR**(-2*self.H) - self.qS**(-2*self.H))

    def correct_prefactor_by_rms_slope(self, value: float):
        self.C0 = 4 * np.pi * (1 - self.H) * value ** 2 / (-0.5 * (1 + self.H) * self.qR**(2-2*self.H)
                                                           + self.qS**(2-2*self.H))

    def generate_height_profile(self, grid: Grid, seed: int | None = None):
        """
        Generates a height profile over the spatial domain specified by the input
        grid, based on spectral properties.

        The method takes a `Grid` object, constructs its spectral mesh, computes
        the wavevector, and maps it to an isotropic power spectral density (PSD).
        The PSD is then converted into a height profile using the provided grid's
        domain lengths and an optional random seed.

        Parameters
        ----------
        grid : Grid
            An object representing the spatial domain over which the height profile
            is to be generated. This object is expected to provide methods for
            forming a spectral mesh and specifying domain lengths.

        seed : int or None, optional
            A random seed for reproducibility of the height profile generation.
            If `None`, the random generation will not be seeded.

        Returns
        -------
        numpy.ndarray
            A 2D array representing the generated height profile based on the
            given grid configuration and spectral properties.
        """
        qx, qy = grid.form_spectral_mesh()
        wavevector = np.stack([qx, qy], axis=0)
        psd = self.mapto_isotropic_psd(wavevector, component_axis=0)
        height = psd_to_height(psd, lateral_sizes=grid.domain_lengths, seed=seed)
        return height


def psd_to_height(psd: np.ndarray, lateral_sizes: Sequence[int] | None = None, seed: int | None = None,
                  random_amplitude: bool=False):
    """
    Convert a power spectral density (PSD) to a height distribution in real space.

    This function takes a 2D power spectral density array and reconstructs the height
    distribution in real space by applying an inverse Fourier transform. The process includes
    scaling the spectral density to amplitude, introducing random phase angles for spatial
    variability, and normalizing the output correctly.

    Parameters
    ----------
    psd : numpy.ndarray
        A 2D array representing the power spectral density.
    lateral_sizes : Sequence[int], optional
        A sequence representing the lateral sizes of the domain in each dimension.
        If None, default sizes of ones are used.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
    random_amplitude : bool, optional
        Whether to impose randomness on the amplitude of the height distribution.

    Returns
    -------
    numpy.ndarray
        A 2D array representing the height distribution in real space.
    """
    if psd.ndim != 2:
        raise ValueError("psd must be a 2D array")

    if lateral_sizes is None:
        lateral_sizes = np.ones(psd.ndim)
    spatial_area = np.multiply.reduce(lateral_sizes)

    # Amplitude
    amplitude = np.sqrt(psd * spatial_area)
    # amplitude = np.sqrt(psd * spatial_area * 2)

    # Impose randomness on the amplitude if required
    if random_amplitude:
        rng = random.default_rng(seed)
        amplitude *= abs(rng.chisquare(2, psd.shape))

    # Impose some random phase angle following uniform distribution
    phasor = generate_phasor_2D_random(psd.shape, seed)

    # Transform back to real space with normalization
    # Set norm="forward" so that NumPy's ifft2 don't do normalization
    return fft.ifft2(amplitude * phasor, norm="forward").real / spatial_area


def generate_phasor_2D_random(shape, seed=None):
    """
    Generate random phase angles for 2D power spectral density.

    Parameters
    ----------
    shape : tuple of int
        A tuple of two integers (nx, ny) representing the shape of the 2D array.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    numpy.ndarray
        A 2D complex array representing the phase_angle with random phase angles.
    """
    nx, ny = shape
    phase_angle = np.empty((nx, ny), dtype=np.float64)
    # Seeded RNG for reproducibility
    rng = random.default_rng(seed)

    # Random phase angle following uniform distribution for half of the spectrum
    phase_angle[:, 0:ny // 2 + 1] = rng.uniform(-np.pi, np.pi, (nx, ny // 2 + 1))
    # The other half is mirrored because real signal has symmetric phase spectrum
    phase_angle[:, -1:ny // 2:-1] = -phase_angle[:, 1:ny // 2 + ny % 2]
    # Phase angles at zero and Nyquist frequency must be 0 (they mirror to themselves)
    phase_angle[0, 0] = 0
    if nx % 2 == 0:
        phase_angle[nx // 2, 0] = 0
    if ny % 2 == 0:
        phase_angle[0, ny // 2] = 0
    if nx % 2 == 0 and ny % 2 == 0:
        phase_angle[nx // 2, ny // 2] = 0

    return np.exp(1j * phase_angle)
