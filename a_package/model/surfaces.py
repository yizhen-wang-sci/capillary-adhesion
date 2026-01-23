"""
Surface geometry generators.

Each generator creates a height field on a given grid based on surface parameters.
This module is independent of config - it only works with primitives (shape string + params).
"""

import dataclasses as dc

import numpy as np
import numpy.linalg as la
import numpy.fft as fft
import numpy.random as random

from a_package.domain import Grid, Field, field_component_ax


# Registry of surface generators
_generators: dict[str, callable] = {}


def _register(shape: str):
    """Decorator to register a surface generator."""
    def decorator(func):
        _generators[shape] = func
        return func
    return decorator


@dc.dataclass(init=True)
class SelfAffineRoughness:
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
        Get the isotropic power spectral density (psd) of a given wavenumber
        - q: wavenumber in radius, i.e. 2 pi over wavelength
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
    # <h^2> corresponding to <PSD>, thus, take the square-root to match overall amplitude
    h_amp = np.sqrt(psd)

    # impose some random phase angle following uniform distribution
    if rng is None:
        rng = random.default_rng(seed)
    phase_angle = np.exp(1j * rng.uniform(0, 2 * np.pi, psd.shape))

    # only the sinusoidal is needed
    return fft.ifft2(h_amp * phase_angle).real


def generate_surface(grid: Grid, shape: str, **params) -> np.ndarray:
    """
    Generate a surface height field based on shape and parameters.

    Parameters
    ----------
    grid : Grid
        The computational grid.
    shape : str
        Surface type identifier ("flat", "tip", "sinusoid", "rough", "pattern").
    **params
        Shape-specific parameters.

    Returns
    -------
    np.ndarray
        Height field array with shape matching grid.nb_elements.

    Examples
    --------
    >>> generate_surface(grid, "flat", constant=0.0)
    >>> generate_surface(grid, "tip", radius=10.0)
    >>> generate_surface(grid, "sinusoid", wavenumber=2.0, amplitude=0.1)
    """
    if shape not in _generators:
        available = list(_generators.keys())
        raise ValueError(f"Unknown surface shape: {shape}. Available: {available}")

    return _generators[shape](grid, **params)


@_register("flat")
def _generate_flat(grid: Grid, constant: float = 0.0) -> np.ndarray:
    """Generate a flat surface at constant height."""
    return constant * np.ones(grid.nb_elements)


@_register("tip")
def _generate_tip(grid: Grid, radius: float) -> np.ndarray:
    """Generate a spherical tip (paraboloid approximation)."""
    R = radius
    [lx, ly] = grid.lengths
    x_center = 0.5 * lx
    y_center = 0.5 * ly
    [x, y] = grid.form_nodal_mesh()
    height = -np.sqrt(np.clip(R**2 - (x - x_center)**2 - (y - y_center)**2, 0, None))
    # Set lowest point to zero
    height += np.amax(abs(height))
    return height


@_register("sinusoid")
def _generate_sinusoid(grid: Grid, wavenumber: float, amplitude: float) -> np.ndarray:
    """Generate a sinusoidal surface."""
    [x, y] = grid.form_nodal_mesh()
    [qx, qy] = grid.form_spectral_mesh()
    height = amplitude * np.cos(qx * x) * np.cos(qy * y)
    return height


@_register("rough")
def _generate_rough(
    grid: Grid,
    prefactor: float,
    rolloff_wavelength_pixels: int,
    cutoff_wavelength_pixels: int,
    hurst_exponent: float,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a self-affine rough surface from PSD."""
    qR = (2 * np.pi) / (grid.element_sizes[0] * rolloff_wavelength_pixels)
    qS = (2 * np.pi) / (grid.element_sizes[0] * cutoff_wavelength_pixels)

    roughness = SelfAffineRoughness(prefactor, qR, qS, hurst_exponent)
    q_2D = grid.form_spectral_mesh()
    [_, C_2D] = roughness.mapto_isotropic_psd(q_2D)

    # Get or generate the seed
    seq = random.SeedSequence(seed)

    # Generate rough surface from PSD
    rng = random.default_rng(seq)
    height = psd_to_height(C_2D, rng=rng)
    return height.squeeze(axis=0)


@_register("pattern")
def _generate_pattern(
    grid: Grid,
    tip_center_x: float = 0.0,
    tip_center_y: float = 0.0,
    wave_len_L: float | None = None,
    wave_amp_L: float | None = None,
    wave_len_M: float | None = None,
    wave_amp_M: float | None = None,
    wave_len_S: float | None = None,
    wave_amp_S: float | None = None,
) -> np.ndarray:
    """Generate a multi-scale wave pattern surface."""
    [xm, ym] = grid.form_nodal_mesh()
    x0 = tip_center_x
    y0 = tip_center_y

    height = np.zeros_like(xm)

    # Large wavelength component
    if wave_len_L is not None and wave_amp_L is not None:
        height += wave_amp_L * \
                  np.cos(2 * np.pi / wave_len_L * (xm - x0)) * \
                  np.cos(2 * np.pi / wave_len_L * (ym - y0))

    # Medium wavelength component
    if wave_len_M is not None and wave_amp_M is not None:
        height += wave_amp_M * \
                  np.cos(2 * np.pi / wave_len_M * (xm - x0)) * \
                  np.cos(2 * np.pi / wave_len_M * (ym - y0))

    # Small wavelength component
    if wave_len_S is not None and wave_amp_S is not None:
        height += wave_amp_S * \
                  np.cos(2 * np.pi / wave_len_S * (xm - x0)) * \
                  np.cos(2 * np.pi / wave_len_S * (ym - y0))

    return np.atleast_2d(height)
