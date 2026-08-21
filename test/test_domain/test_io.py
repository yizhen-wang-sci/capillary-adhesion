"""
Tests of the `storing.py` file.
"""
import numpy as np
import pytest

from NuMPI.Testing.Assertions import assert_all_array_equal

from a_package.domain.io import NpyIO
from test.test_domain.utils import generate_global_random_field


@pytest.fixture
def mock_field(mock_grid, comm_world):
    return generate_global_random_field(mock_grid.nb_domain_grid_pts, comm_world)


@pytest.fixture
def mock_array(comm_world):
    array = np.empty(10, dtype=float)
    if comm_world.rank == 0:
        rng = np.random.default_rng()
        array[...] = rng.random(array.shape)
    comm_world.Bcast(array, root=0)
    return array


def test_save_load_distributed(mpi_tmp_path, decomposed_grid, mock_field, comm_world):
    decomposition = decomposed_grid.decomposition
    io = NpyIO(mpi_tmp_path, decomposition, communicator=comm_world)
    name = "test_distributed"

    io.save_distributed(name, decomposed_grid.get_local(mock_field))
    loaded_arr = io.load_distributed(name)
    assert_all_array_equal(comm_world, loaded_arr, decomposed_grid.get_local(mock_field))


def test_save_load_singular(mpi_tmp_path, decomposed_grid, mock_array, comm_world):
    io = NpyIO(mpi_tmp_path, decomposed_grid.decomposition, communicator=comm_world)
    name = "test_singular"

    io.save_singular(name, mock_array)
    loaded_arr = io.load_singular(name)
    if comm_world.Get_rank() == 0:
        np.testing.assert_equal(loaded_arr, mock_array)
    else:
        assert loaded_arr is None


def test_load_replicated(mpi_tmp_path, decomposed_grid, mock_array, comm_world):
    io = NpyIO(mpi_tmp_path, decomposed_grid.decomposition, communicator=comm_world)
    name = "test_replicated"

    io.save_singular(name, mock_array)
    loaded = io.load_replicated(name)
    np.testing.assert_equal(loaded, mock_array)


def test_load_singular_missing_file(mpi_tmp_path, decomposed_grid, comm_world):
    io = NpyIO(mpi_tmp_path, decomposed_grid.decomposition, communicator=comm_world)
    with pytest.raises(FileNotFoundError):
        io.load_singular("non_existent")


def test_load_replicated_missing_file(mpi_tmp_path, decomposed_grid, comm_world):
    io = NpyIO(mpi_tmp_path, decomposed_grid.decomposition, communicator=comm_world)
    with pytest.raises(FileNotFoundError):
        io.load_replicated("non_existent")


def test_undecomposed_store_round_trips(mpi_tmp_path, mock_array, comm_world):
    io = NpyIO(mpi_tmp_path)
    if comm_world.rank == 0:
        io.save_distributed("undecomposed", mock_array)
        np.testing.assert_equal(io.load_distributed("undecomposed"), mock_array)


@pytest.mark.parametrize(
    ("load_name", "broken_on_root"),
    [
        ("load_distributed", True),
        ("load_distributed", False),
        ("load_singular", True),
        ("load_replicated", True),
        ("load_replicated", False),
    ],
)
def test_load_raises_on_every_rank(
    load_name, broken_on_root, mpi_tmp_path, decomposed_grid, mock_field, mock_array, comm_world
):
    """A file missing for the rank that reads it."""
    decomposition = decomposed_grid.decomposition
    store = NpyIO(mpi_tmp_path, decomposition, communicator=comm_world)
    store.save_distributed("field", decomposed_grid.get_local(mock_field))
    store.save_singular("array", mock_array)
    comm_world.Barrier()

    empty = mpi_tmp_path / "empty"
    if comm_world.rank == 0:
        empty.mkdir(exist_ok=True)
    comm_world.Barrier()

    is_broken = (comm_world.rank == 0) == broken_on_root
    root_path = empty if is_broken else mpi_tmp_path
    io = NpyIO(root_path, decomposition, communicator=comm_world)
    load = getattr(io, load_name)
    name = "field" if load_name == "load_distributed" else "array"

    if broken_on_root or comm_world.Get_size() > 1:
        with pytest.raises(FileNotFoundError):
            load(name)
    else:
        load(name)
