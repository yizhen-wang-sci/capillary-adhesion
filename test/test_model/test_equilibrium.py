"""Tests of the equilibrium formulations."""

import numpy as np
import pytest

from a_package.domain.grid import Grid, factorize_closest
from a_package.model import (
    CapillaryBridge,
    PhaseMixture,
    formulate_constant_pressure_phase_problem,
    formulate_constant_volume_phase_problem,
)


@pytest.fixture
def capillary(comm_world):
    grid = Grid([4, 4], [4.0, 4.0])
    grid.decompose(factorize_closest(comm_world.Get_size(), 2), (1, 1), communicator=comm_world)
    capillary = CapillaryBridge(grid, PhaseMixture(eta=1.0, theta=np.pi / 3), communicator=comm_world)
    capillary.set_gap(np.ones(grid.decomposition.nb_subdomain_grid_pts))
    capillary.set_phase(np.full(grid.decomposition.nb_subdomain_grid_pts, 0.5))
    return capillary


# =============================================================================
# Constant volume
# =============================================================================


def test_constant_volume_problem_carries_the_linear_constraint(capillary):
    problem = formulate_constant_volume_phase_problem(capillary, volume=2.0)
    assert problem.has_linear_constraints
    assert not problem.has_equality_constraints
    assert problem.b == pytest.approx(2.0)


def test_constant_volume_problem_objective_is_the_energy(capillary):
    problem = formulate_constant_volume_phase_problem(capillary, volume=2.0)
    assert problem.get_f() == pytest.approx(capillary.get_energy())
    np.testing.assert_allclose(problem.get_f_Dx(), capillary.get_energy_jacobian().ravel())


# =============================================================================
# Constant pressure
# =============================================================================


def test_constant_pressure_problem_objective_is_the_helmholtz_potential(capillary):
    pressure = 0.5
    problem = formulate_constant_pressure_phase_problem(capillary, pressure=pressure)
    expected = capillary.get_energy() - pressure * capillary.get_volume()
    assert problem.get_f() == pytest.approx(expected)


def test_constant_pressure_problem_jacobian_carries_the_pressure_term(capillary):
    pressure = 0.5
    problem = formulate_constant_pressure_phase_problem(capillary, pressure=pressure)
    expected = capillary.get_energy_jacobian() - pressure * capillary.get_volume_jacobian()
    np.testing.assert_allclose(problem.get_f_Dx(), expected.ravel())
