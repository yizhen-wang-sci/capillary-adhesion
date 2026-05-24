import pytest

import numpy as np
from NuMPI import MPI

from a_package.domain.quadrature import Quadrature, centroid_quadrature, three_pt_quadrature

from test.test_domain.utils import generate_global_random_field


nodal_quadrature = Quadrature([[0., 0.]], [1.])


@pytest.fixture(params=[nodal_quadrature, centroid_quadrature, three_pt_quadrature])
def mock_quadrature(request):
    return request.param


def test_integral(decomposed_grid, mock_quadrature, comm_world):
    field = generate_global_random_field(decomposed_grid.nb_domain_grid_pts, comm_world)
    local_field = np.expand_dims(field[*decomposed_grid.decomposition.icoords], axis=(0, 1))
    value = mock_quadrature.integrate(local_field)
    value = comm_world.allreduce(value, op=MPI.SUM)
    expected_value = np.sum(field)

    assert np.allclose(value, expected_value, atol=1e-12)


def test_integral_perturbed(decomposed_grid, mock_quadrature, comm_world):
    field = generate_global_random_field(decomposed_grid.nb_domain_grid_pts, comm_world)
    perturb = 1e-3
    for idx in np.ndindex(decomposed_grid.nb_domain_grid_pts):
        field[idx] += perturb
        local_field = np.expand_dims(field[*decomposed_grid.decomposition.icoords], axis=(0, 1))
        value = mock_quadrature.integrate(local_field)
        value = comm_world.allreduce(value, op=MPI.SUM)
        expected_value = np.sum(field)
        assert np.allclose(value, expected_value, atol=1e-12)
        field[idx] -= perturb
