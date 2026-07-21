"""
Tests of the self-affine roughness model.
"""

import numpy as np
import pytest

from a_package.domain.grid import Grid
from a_package.model.roughness import SelfAffineRoughness, psd_to_height, generate_phasor_2D_random


@pytest.fixture
def roughness():
    return SelfAffineRoughness(C0=1.0, H=0.8, qR=(2 * np.pi) * 64, qS=(2 * np.pi) * 256)


@pytest.fixture
def large_grid():
    nb_grid_points = 512
    return Grid([nb_grid_points, nb_grid_points], [1., 1.])


def compute_height_variance(roughness) -> float:
    wavenumber = np.concatenate([
        # Constant part: C q is still linear, only requires 2 points
        np.linspace(roughness.qT, roughness.qR, 2),
        np.linspace(roughness.qR, roughness.qS, 200)])
    return np.trapezoid(wavenumber * roughness.mapto_isotropic_psd(wavenumber), wavenumber) / (2 * np.pi)

def compute_slope_variance(roughness) -> float:
    wavenumber = np.concatenate([
        np.linspace(roughness.qT, roughness.qR, 200),
        np.linspace(roughness.qR, roughness.qS, 200)])
    return np.trapezoid(wavenumber ** 3 * roughness.mapto_isotropic_psd(wavenumber), wavenumber) / (2 * np.pi)


def test_psd_at_zero_is_zero(roughness):
    """The PSD evaluated at wavenumber 0 must be 0 (below qT)."""
    q = np.linspace(0., 9., 10)
    psd = roughness.mapto_isotropic_psd(q)
    assert psd[np.isclose(q, 0)] == 0.0


def test_psd_to_height_has_zero_mean(large_grid, roughness):
    """The height field obtained via psd_to_height should have zero mean."""
    wavevector = large_grid.form_spectral_mesh()
    psd = roughness.mapto_isotropic_psd(wavevector, component_axis=0)
    height = psd_to_height(psd, seed=None)

    assert np.isclose(np.mean(height), 0.0, atol=1e-12)


@pytest.mark.parametrize("shape", [(8, 8), (9, 9), (8, 9), (9, 8)],
                         ids=["even-even", "odd-odd", "even-odd", "odd-even"])
def test_generate_phasor_2D_random_hermitian(shape):
    """A real signal requires phase(-k) = -phase(k) (indices modulo the grid shape)."""
    phasor = generate_phasor_2D_random(shape, seed=0)
    phase = np.angle(phasor)
    nx, ny = shape

    i = np.arange(nx)[:, None]
    j = np.arange(ny)[None, :]
    mirrored_phase = phase[(-i) % nx, (-j) % ny]

    # Wrap into [-pi, pi) before comparing to zero, since phase angles are only
    # defined modulo 2*pi.
    residual = (phase + mirrored_phase + np.pi) % (2 * np.pi) - np.pi
    broken = np.argwhere(~np.isclose(residual, 0.0, atol=1e-9))

    assert broken.size == 0, (
        f"Hermitian symmetry phase[-i, -j] == -phase[i, j] broken at (row, col) "
        f"indices: {broken.tolist()}"
    )


def test_roughness_correct_prefactor_by_rms_height(roughness):
    h_rms_specified = 1.
    roughness.correct_prefactor_by_rms_height(h_rms_specified)
    assert np.isclose(compute_height_variance(roughness), h_rms_specified**2, atol=2e-4)


def test_psd_to_height_normalization_with_rms_height(large_grid, roughness):
    h_rms_specified = 1.
    roughness.correct_prefactor_by_rms_height(h_rms_specified)
    height = roughness.generate_height_profile(large_grid)
    assert np.isclose(np.var(height), h_rms_specified**2)


def test_roughness_correct_prefactor_by_rms_slope(roughness):
    slope_rms_specified = 1.
    roughness.correct_prefactor_by_rms_slope(slope_rms_specified)
    assert np.isclose(compute_slope_variance(roughness), slope_rms_specified**2, atol=2e-3)


def test_psd_to_height_normalization_with_rms_slope(large_grid, roughness):
    rms_slope_specified = 1.
    roughness.correct_prefactor_by_rms_slope(rms_slope_specified)
    height = roughness.generate_height_profile(large_grid)
    hx, hy = np.gradient(height, *large_grid.element_sizes)
    slope_variance = np.sum(hx**2 + hy**2) / np.multiply.reduce(large_grid.nb_domain_grid_pts)
    assert np.isclose(slope_variance, rms_slope_specified**2)
