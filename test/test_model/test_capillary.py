"""
Tests of the capillary model.
"""

import math
import sys

import numpy as np
import matplotlib.pyplot as plt
import pytest

from a_package.domain import Grid, adapt_shape
from a_package.model.capillary import CapillaryBridge, NodalFormCapillary


show_me_plot = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_fields():
    """Create test fields: gap (sphere on flat) and phase (circular pattern)."""
    L = 4.0
    N = 4
    shape = (1, 1, N, N)

    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    xm, ym = np.meshgrid(x, y, indexing='xy')

    # Gap: sphere on flat
    R = 10.0
    h1 = -np.sqrt(np.clip(R**2 - (xm - 0.5*L)**2 - (ym - 0.5*L)**2, 0, np.inf))
    h1 = h1 - h1.min()
    gap = np.clip(h1 - 0.1, 0, None).reshape(shape)

    # Phase: circular pattern
    mask = (xm/L)**2 + (ym/L)**2 >= 0.5**2
    phase = np.ones_like(xm)
    phase[mask] = 0.0
    phase = phase.reshape(shape)

    # Phase gradient (arbitrary nonzero values)
    phase_grad = np.stack([
        0.1 * np.sin(2 * np.pi * xm / L),
        0.1 * np.cos(2 * np.pi * ym / L),
    ], axis=0).reshape(2, 1, N, N)

    return {
        "L": L,
        "N": N,
        "xm": xm,
        "ym": ym,
        "gap": gap,
        "phase": phase,
        "phase_grad": phase_grad,
    }


# =============================================================================
# Utilities
# =============================================================================


def compute_numerical_jacobian(x, func, deltas=None):
    """
    Compute numerical jacobian using central finite differences.

    Parameters
    ----------
    x : np.ndarray
        Input array to differentiate with respect to.
    func : callable
        Function that takes x and returns a scalar.
    deltas : np.ndarray, optional
        Step sizes to use. If None, uses range from machine precision to 1.

    Returns
    -------
    numeric_jacobian : np.ndarray
        Shape (len(deltas), *x.shape), jacobian for each step size.
    deltas : np.ndarray
        Step sizes used.
    """
    if deltas is None:
        lowest_magnitude = math.floor(0.5 * math.log10(sys.float_info.epsilon))
        highest_magnitude = 1.0
        deltas = np.pow(10.0, np.arange(lowest_magnitude, highest_magnitude))

    numeric_jacobian = np.empty((deltas.size, *x.shape))
    for i, delta in enumerate(deltas):
        for indices in np.ndindex(x.shape):
            original = x[indices].copy()
            x[indices] = original + delta
            plus_val = func(x)
            x[indices] = original - delta
            minus_val = func(x)
            numeric_jacobian[(i, *indices)] = 0.5 * (plus_val - minus_val) / delta
            x[indices] = original

    return numeric_jacobian, deltas


def assert_jacobian_correct(impl_jacobian, numeric_jacobian, deltas, tol=1e-6, show_plot=False):
    """
    Assert that implementation jacobian matches numerical jacobian.

    Compares across all step sizes and checks that minimum difference is below tolerance.
    """
    jac_diffs = np.abs(impl_jacobian - numeric_jacobian).squeeze()
    diffs = np.max(jac_diffs, axis=(-2, -1))

    if show_plot:
        plt.plot(deltas, diffs, "x-", label=r"Finite difference error")
        plt.loglog()
        plt.xlabel(r"$\delta$")
        plt.ylabel(r"$\varepsilon$")
        plt.legend()
        plt.show()

    assert np.min(diffs) < tol, f"Jacobian difference {np.min(diffs):.2e} exceeds tolerance {tol:.2e}"


# =============================================================================
# Tests
# =============================================================================


def test_energy_jacobian_in_model(test_fields):
    """Test CapillaryBridge.compute_local_energy_jacobian against finite differences."""
    eta = 1.0
    theta = np.pi / 3

    bridge = CapillaryBridge(eta=eta, theta=theta)
    gap = test_fields["gap"]
    phase = test_fields["phase"].copy()
    phase_grad = test_fields["phase_grad"]

    def total_energy(ph):
        return np.sum(bridge.compute_local_energy(gap, ph, phase_grad))

    numeric_jacobian, deltas = compute_numerical_jacobian(phase, total_energy)
    impl_jacobian, _ = bridge.compute_local_energy_jacobian(gap, phase, phase_grad)

    assert_jacobian_correct(impl_jacobian, numeric_jacobian, deltas, show_plot=show_me_plot)


def test_energy_jacobian_in_formulation(test_fields):
    """Test NodalFormCapillary.get_energy_jacobian against finite differences."""
    L = test_fields["L"]
    N = test_fields["N"]
    eta = 1.0
    theta = np.pi / 3

    grid = Grid([L, L], [N, N])
    capillary = NodalFormCapillary(grid, {"theta": theta, "eta": eta})
    capillary.set_gap(test_fields["gap"].squeeze())

    phase = adapt_shape(test_fields["phase"].squeeze())

    def energy_func(ph):
        capillary.set_phase(ph)
        return capillary.get_energy()

    numeric_jacobian, deltas = compute_numerical_jacobian(phase, energy_func)

    capillary.set_phase(phase)
    impl_jacobian = capillary.get_energy_jacobian()

    assert_jacobian_correct(impl_jacobian, numeric_jacobian, deltas, show_plot=show_me_plot)
