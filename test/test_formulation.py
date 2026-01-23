"""
Tests of the `modelling.py` file.
"""

import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from a_package.domain import Grid, adapt_shape
from a_package.model.capillary import NodalFormCapillary


show_me_plot = False


def test_energy_jacobian_in_formulation():
    # Formulate the model
    a = 1.0
    L = 4.0
    N = 4
    grid = Grid([L, L], [N, N])

    eta = 1. * a
    theta = np.pi / 3

    [xm, ym] = grid.form_nodal_mesh()

    # the upper surface is spherical
    R = 10.0
    h1 = -np.sqrt(np.clip(R**2 - (xm - 0.5*L)**2 - (ym - 0.5*L)**2, 0, np.inf))
    # set the minimum to zero
    h1 = h1 - h1.min()
    # the lower surafce is flat
    h2 = np.zeros_like(h1)

    # make sure there are some areas in contact
    gap = np.clip(h1 -0.1 * a -h2, 0, None)

    capillary = NodalFormCapillary(grid, {"theta": theta, "eta": eta})
    capillary.set_gap(gap)

    # All the step lengths to be used for finite difference computation
    lowest_magnitude = math.floor(0.5 * math.log10(sys.float_info.epsilon))  # machine precision determined
    highest_magnitude = 1.0
    deltas = np.pow(10.0, np.arange(lowest_magnitude, highest_magnitude))

    # A circular phase field for testing
    phi = np.ones_like(h1)
    phi[(xm/L)**2 + (ym/L)**2 >= 0.5**2] = 0.0
    phi = adapt_shape(phi)

    # Compute jacobian numerically (2-order finite difference)
    numeric_jacobian = np.empty((deltas.size, *phi.shape))
    for i, delta in enumerate(deltas):
        for indices in np.ndindex(phi.shape):
            # backup the original value
            original_value = np.copy(phi[tuple(indices)])
            # get numerical jacobian via central difference
            phi[tuple(indices)] = original_value + delta
            capillary.set_phase(phi)
            plus_val = capillary.get_energy()
            phi[tuple(indices)] = original_value - delta
            capillary.set_phase(phi)
            minus_val = capillary.get_energy()
            numeric_jacobian[(i, *indices)] = 0.5 * (plus_val - minus_val) / delta
            # recover the original value
            phi[tuple(indices)] = original_value

    # Compute jacobian from the implementation
    capillary.set_phase(phi)
    impl_jacobian = capillary.get_energy_jacobian()

    # Measure the difference
    jacobian_diffs = abs(impl_jacobian - numeric_jacobian).squeeze()
    diffs = np.max(jacobian_diffs, axis=(-2,-1))

    # Plots
    if show_me_plot:
        plt.plot(deltas, diffs , "x-",
                 label=r"Difference from a numerical method of $\mathcal{O}(\delta^2)$")

        plt.loglog()
        plt.xlabel(r"$\delta$")
        plt.ylabel(r"$\varepsilon$")
        plt.legend()

        plt.show()

    # Assertion
    eps = 1e-6
    assert np.min(diffs) < eps, f"The difference exceeds the tolerance {eps:.2e}"
