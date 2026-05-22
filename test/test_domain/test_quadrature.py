import pytest

import numpy as np
from NuMPI import MPI

from a_package.domain.grid import Grid
from a_package.domain.quadrature import Quadrature


@pytest.fixture
def small_grid(comm_world):
    grid = Grid((4, 4))
    grid.decompose((comm_world.Get_size(), 1), communicator=comm_world)
    return grid


@pytest.fixture
def mock_field(small_grid):
    return np.ones(small_grid.nb_domain_grid_pts)


@pytest.fixture
def mock_quadrature():
    return Quadrature([[0., 0.]], [1.])


def slice_subdomain(grid, data):
    """Return the subdomain slice of the data. The spatial dimensions are assumed the last ones."""
    return data[..., *grid.decomposition.icoords]


def test_integral(small_grid, mock_field, mock_quadrature, comm_world):
    local_field = np.expand_dims(slice_subdomain(small_grid, mock_field), axis=(0, 1))
    value = mock_quadrature.integrate(local_field)
    value = comm_world.allreduce(value, op=MPI.SUM)
    expected_value = np.sum(mock_field)
    print(f"Value: {value}, Expected value: {expected_value}")
    assert np.allclose(value, expected_value, atol=1e-12)


def test_integral_perturbed(small_grid, mock_field, mock_quadrature, comm_world):
    perturb = 1e-3
    for idx in np.ndindex(small_grid.nb_domain_grid_pts):
        mock_field[idx] += perturb
        local_field = np.expand_dims(slice_subdomain(small_grid, mock_field), axis=(0, 1))
        value = mock_quadrature.integrate(local_field)
        value = comm_world.allreduce(value, op=MPI.SUM)
        expected_value = np.sum(mock_field)
        print(f"Value: {value}, Expected value: {expected_value}")
        assert np.allclose(value, expected_value, atol=1e-12)
        mock_field[idx] -= perturb
