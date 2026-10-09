import numpy as np

from test.utils import generate_global_random_field, generate_global_range_field


def test_real_field_decomposition(mock_grid, decompose_stitch, comm_world):
    # a global field
    mock_field = generate_global_random_field(mock_grid.nb_domain_grid_pts, comm_world)

    # decompose and stitch
    decompose, stitch = decompose_stitch
    decompose(mock_grid)
    field = mock_grid.collection_real.real_field("test_field", 1)
    field.s[0, 0, ...] = mock_grid.get_local(mock_field)
    collected = stitch(field.s[0, 0, ...], mock_grid)

    # assertions
    assert np.allclose(collected, mock_field)


def test_int_field_decomposition(mock_grid, decompose_stitch, comm_world):
    # A global field
    mock_field = generate_global_range_field(mock_grid.nb_domain_grid_pts, comm_world)

    # decompose and stitch
    decompose, stitch = decompose_stitch
    decompose(mock_grid)
    field = mock_grid.collection_real.int_field("test_field", 1)
    field.s[0, 0, ...] = mock_grid.get_local(mock_field)
    collected = stitch(field.s[0, 0, ...], mock_grid)

    # assertions
    assert np.allclose(collected, mock_field)
