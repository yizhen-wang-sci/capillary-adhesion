import pytest
import numpy as np

from test.test_domain.utils import generate_global_random_field, generate_global_range_field


def test_real_field_decomposition(mock_grid, decompose_stitch, comm_world):
    # a global field
    mock_field = generate_global_random_field(mock_grid.nb_domain_grid_pts, comm_world)

    # decompose and stitch
    decompose, stitch = decompose_stitch
    decomposition = decompose(mock_grid)
    field = decomposition.collection.real_field("test_field", 1)
    field.s[0, 0, ...] = mock_field[*decomposition.icoords]
    collected = stitch(field.s[0, 0, ...], mock_grid)

    # assertions
    assert np.allclose(collected, mock_field)


def test_int_field_decomposition(mock_grid, decompose_stitch, comm_world):
    # A global field
    mock_field = generate_global_range_field(mock_grid.nb_domain_grid_pts, comm_world)

    # decompose and stitch
    decompose, stitch = decompose_stitch
    decomposition = decompose(mock_grid)
    field = decomposition.collection.int_field("test_field", 1)
    field.s[0, 0, ...] = mock_field[*decomposition.icoords]
    collected = stitch(field.s[0, 0, ...], mock_grid)

    # assertions
    assert np.allclose(collected, mock_field)
