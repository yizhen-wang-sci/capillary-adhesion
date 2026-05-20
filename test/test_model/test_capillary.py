"""
Tests of the capillary model.
"""

import math
import sys

import numpy as np
import matplotlib.pyplot as plt
import pytest

from a_package.domain import Grid
from a_package.model.capillary import PhaseMixture, CapillaryBridge


show_me_plot = False


@pytest.fixture
def test_grid():
    return Grid([10, 10], [2., 2.])


@pytest.fixture
def test_phase_mixture():
    return PhaseMixture(eta=1.0, theta=np.pi / 3)


@pytest.fixture
def sphere_flat(test_grid):
    R = 10 * np.sqrt(np.sum(np.square(test_grid.element_sizes)))
    Lx, Ly = test_grid.domain_lengths
    xm, ym = test_grid.form_spatial_mesh()
    h = -np.sqrt(np.clip(R**2 - (xm - 0.5*Lx)**2 - (ym - 0.5*Ly)**2, 0, np.inf))
    # Set the tip center as the unit height.
    return h - h.min() + 1


@pytest.fixture
def sinusoidal_field(test_grid):
    """A field with continuous values between 0 and 1."""
    xm, ym = test_grid.form_spatial_mesh()
    Lx, Ly = test_grid.domain_lengths
    return 0.5 * np.cos(2 * np.pi * xm / Lx) + 0.5 * np.sin(2 * np.pi * ym / Ly)


@pytest.fixture
def inner_circle_field(test_grid):
    """A field with jumps at 0 / 1 border."""
    xm, ym = test_grid.form_spatial_mesh()
    Lx, Ly = test_grid.domain_lengths
    phase = np.zeros(test_grid.nb_domain_grid_pts)
    inner = (xm/Lx)**2 + (ym/Ly)**2 <= 0.5**2
    phase[inner] = 1.0
    return phase


@pytest.fixture(params=["sinusoidal_field", "inner_circle_field"])
def test_field(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def small_steps():
    # largest step is 10^0 = 1.
    highest_magnitude = 0
    # this ensures the range covers the square root of machine precision
    lowest_magnitude = math.floor(0.5 * math.log10(sys.float_info.epsilon))
    return np.pow(10.0, np.arange(lowest_magnitude, highest_magnitude + 1))


def compute_numerical_jacobian(x, func, step_sizes):
    """
    Compute numerical jacobian using central finite differences.

    Parameters
    ----------
    x : np.ndarray
        Input array to differentiate with respect to.
    func : callable
        Function that takes x and returns a scalar.
    step_sizes : np.ndarray
        Step sizes to use. If None, uses range from machine precision to 1.

    Returns
    -------
    numeric_jacobian : np.ndarray
        Shape (len(deltas), *x.shape), jacobian for each step size.
    """
    numeric_jacobian = np.empty((np.size(step_sizes), *x.shape))
    for i_delta, delta in enumerate(step_sizes):
        for indices in np.ndindex(x.shape):
            original = x[indices].copy()
            x[indices] = original + delta
            plus_val = func(x)
            x[indices] = original - delta
            minus_val = func(x)
            numeric_jacobian[(i_delta, *indices)] = 0.5 * (plus_val - minus_val) / delta
            x[indices] = original
    return numeric_jacobian


def assert_jacobian_correct(impl_jacobian, numeric_jacobian, step_sizes, tol=1e-6, show_plot=False):
    """
    Assert that implementation jacobian matches numerical jacobian.

    Compares across all step sizes and checks that minimum difference is below tolerance.
    """
    jac_diffs = np.abs(impl_jacobian - numeric_jacobian).squeeze()
    diffs = np.max(jac_diffs, axis=(-2, -1))

    if show_plot:
        plt.plot(step_sizes, diffs, "x-", label=r"Finite difference error")
        plt.loglog()
        plt.xlabel(r"$\delta$")
        plt.ylabel(r"$\varepsilon$")
        plt.legend()
        plt.show()

    assert np.min(diffs) < tol, f"Jacobian difference {np.min(diffs):.2e} exceeds tolerance {tol:.2e}"


def test_energy_jacobian(test_grid, test_phase_mixture, sphere_flat, test_field, small_steps):
    """Test NodalFormCapillary.get_energy_jacobian against finite differences."""
    capillary = CapillaryBridge(test_grid, test_phase_mixture)
    capillary.set_gap(sphere_flat)

    def energy_func(phase):
        capillary.set_phase(phase)
        return capillary.get_energy()

    numeric_jacobian = compute_numerical_jacobian(test_field, energy_func, small_steps)

    capillary.set_phase(test_field)
    impl_jacobian = capillary.get_energy_jacobian()

    assert_jacobian_correct(impl_jacobian, numeric_jacobian, small_steps, show_plot=show_me_plot)


def test_volume_jacobian(test_grid, test_phase_mixture, sphere_flat, test_field, small_steps):
    """Test NodalFormCapillary.get_energy_jacobian against finite differences."""
    capillary = CapillaryBridge(test_grid, test_phase_mixture)
    capillary.set_gap(sphere_flat)

    def volume_func(phase):
        capillary.set_phase(phase)
        return capillary.get_volume()

    numeric_jacobian = compute_numerical_jacobian(test_field, volume_func, small_steps)

    capillary.set_phase(test_field)
    impl_jacobian = capillary.get_volume_jacobian()

    assert_jacobian_correct(impl_jacobian, numeric_jacobian, small_steps, show_plot=show_me_plot)
