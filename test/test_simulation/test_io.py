"""
Tests for SimulationIO.
"""

import pytest
import numpy as np
from NuMPI import MPI

from a_package.domain import Grid, factorize_closest
from a_package.simulation.io import SimulationIO


@pytest.fixture
def comm_world():
    return MPI.COMM_WORLD


@pytest.fixture
def mpi_tmp_path(tmp_path_factory, comm_world):
    if comm_world.rank == 0:
        path = tmp_path_factory.mktemp("mpi")
    else:
        path = None
    path = comm_world.bcast(path, root=0)
    # 'yield' so the script after is executed before test teardown
    yield path
    # prevent faster process from deleting the directory
    comm_world.Barrier()


@pytest.fixture
def grid():
    """Create a simple grid."""
    return Grid((8, 8))


@pytest.fixture
def decomposition(grid, comm_world):
    return grid.decompose(factorize_closest(comm_world.Get_size(), 2), communicator=comm_world)


@pytest.fixture
def io(decomposition, mpi_tmp_path, comm_world):
    """Create a SimulationIO instance."""
    return SimulationIO(mpi_tmp_path, decomposition, communicator=comm_world)


def localize(field, decomposition):
    if decomposition is None:
        return field
    return field[..., *decomposition.icoords]


# =============================================================================
# Constants
# =============================================================================


def test_save_load_constant_field(grid, decomposition, io, comm_world):
    """Save and load constant field."""
    field = comm_world.bcast(np.random.random((1, 1, *grid.nb_domain_grid_pts)))
    expected = localize(field, decomposition)

    io.save_constant(fields={"my_field": expected})
    result = io.load_constant(field_names=["my_field"])

    np.testing.assert_array_almost_equal(result["my_field"], expected)


def test_save_load_constant_single_value(io):
    """Save and load constant single value."""
    io.save_constant(single_values={"my_value": 1.5})
    result = io.load_constant(single_value_names=["my_value"])

    assert result["my_value"] == pytest.approx(1.5)


def test_save_load_constant_mixed(grid, decomposition, io, comm_world):
    """Save and load both fields and single values."""
    field = comm_world.bcast(np.random.random((1, 1, *grid.nb_domain_grid_pts)))
    expected = localize(field, decomposition)

    io.save_constant(fields={"field_a": expected}, single_values={"value_a": 2.0})
    result = io.load_constant(field_names=["field_a"], single_value_names=["value_a"])

    np.testing.assert_array_almost_equal(result["field_a"], expected)
    assert result["value_a"] == pytest.approx(2.0)


# =============================================================================
# Steps
# =============================================================================


def test_save_load_step_field(grid, decomposition, io, comm_world):
    """Save and load step field."""
    field = comm_world.bcast(np.random.random((1, 1, *grid.nb_domain_grid_pts)))
    expected = localize(field, decomposition)

    io.save_step(0, fields={"field": expected})
    result = io.load_step(0, field_names=["field"])

    np.testing.assert_array_almost_equal(result["field"], expected)


def test_save_load_step_single_value(io):
    """Save and load step single value."""
    io.save_step(0, single_values={"x": 0.1})
    io.save_step(1, single_values={"x": 0.2})

    result0 = io.load_step(0, single_value_names=["x"])
    result1 = io.load_step(1, single_value_names=["x"])

    assert result0["x"] == pytest.approx(0.1)
    assert result1["x"] == pytest.approx(0.2)


def test_save_load_multiple_steps(grid, decomposition, io, comm_world):
    """Save and load multiple steps."""
    fields = [comm_world.bcast(np.random.random((1, 1, *grid.nb_domain_grid_pts))) for _ in range(3)]
    values = [0.1, 0.2, 0.3]

    for i, (field, val) in enumerate(zip(fields, values)):
        io.save_step(i, fields={"field": localize(field, decomposition)}, single_values={"val": val})

    for i, (field, expected_val) in enumerate(zip(fields, values)):
        result = io.load_step(i, field_names=["field"], single_value_names=["val"])
        np.testing.assert_array_almost_equal(result["field"], localize(field, decomposition))
        assert result["val"] == pytest.approx(expected_val)


# =============================================================================
# Trajectory
# =============================================================================


def test_save_load_trajectory_single_values(io, comm_world):
    """Save and load trajectory single values."""
    arr_a = comm_world.bcast(np.array([0.1, 0.2, 0.3]))
    arr_b = comm_world.bcast(np.array([1.0, 2.0, 3.0]))

    io.save_trajectory(single_values={"a": arr_a, "b": arr_b})
    result = io.load_trajectory(single_value_names=["a", "b"])

    np.testing.assert_array_almost_equal(result["a"], arr_a)
    np.testing.assert_array_almost_equal(result["b"], arr_b)


def test_load_trajectory_fields_lazy(grid, decomposition, io, comm_world):
    """load_trajectory returns lazy-loading FieldArray for fields."""
    fields = [comm_world.bcast(np.random.random((1, 1, *grid.nb_domain_grid_pts))) for _ in range(3)]

    for i, field in enumerate(fields):
        io.save_step(i, fields={"field": localize(field, decomposition)})

    result = io.load_trajectory(field_names=["field"])
    field_array = result["field"]

    for i, field in enumerate(fields):
        np.testing.assert_array_almost_equal(field_array[i], localize(field, decomposition))


def test_save_trajectory_fields(grid, decomposition, io, comm_world):
    """save_trajectory saves fields by index."""
    fields = [comm_world.bcast(np.random.random((1, 1, *grid.nb_domain_grid_pts))) for _ in range(3)]
    localized_fields = [localize(f, decomposition) for f in fields]

    io.save_trajectory(fields={"field": localized_fields})
    result = io.load_trajectory(field_names=["field"])

    for i, expected in enumerate(localized_fields):
        np.testing.assert_array_almost_equal(result["field"][i], expected)
