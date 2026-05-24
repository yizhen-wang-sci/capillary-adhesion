"""
Tests of the `storing.py` file.
"""
import numpy as np
import pytest

from NuMPI.Testing.Assertions import assert_all_array_equal

from a_package.domain.io import NpyIO
from test.test_domain.utils import generate_global_random_field


@pytest.fixture
def mpi_tmp_path(tmp_path_factory, comm_world):
    path = None
    if comm_world.rank == 0:
        path = tmp_path_factory.mktemp("mpi")
    path = comm_world.bcast(path, root=0)
    yield path
    # prevent faster process from deleting the directory before slower ones are done
    comm_world.Barrier()


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

    io.save_distributed(name, mock_field[*decomposition.icoords])
    loaded_arr = io.load_distributed(name)
    assert_all_array_equal(comm_world, loaded_arr, mock_field[*decomposition.icoords])


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
