"""
Tests of the `storing.py` file.
"""
import numpy as np
import pytest

from NuMPI.Testing.Assertions import assert_all_array_equal

from a_package.domain.io import NpyIO
from a_package.domain import Grid, factorize_closest


@pytest.fixture
def field(comm_world):
    return comm_world.bcast(np.random.rand(10, 10))


@pytest.fixture
def array(comm_world):
    return comm_world.bcast(np.random.rand(10))


@pytest.fixture
def mpi_tmp_path(tmp_path_factory, comm_world):
    if comm_world.rank == 0:
        path = tmp_path_factory.mktemp("mpi")
    else:
        path = None
    path = comm_world.bcast(path, root=0)
    yield path
    # prevent faster process from deleting the directory
    comm_world.Barrier()


def test_save_load_distributed(mpi_tmp_path, field, comm_world):
    grid = Grid(field.shape)
    decomposition = grid.decompose(factorize_closest(comm_world.Get_size(), 2), nb_ghost_layers=(1, 1), communicator=comm_world)
    io = NpyIO(mpi_tmp_path, decomposition)
    name = "test_distributed"

    io.save_distributed(name, field[*decomposition.icoords])
    loaded_arr = io.load_distributed(name)
    assert_all_array_equal(comm_world, loaded_arr, field[*decomposition.icoords])


def test_save_load_singular(mpi_tmp_path, array, comm_world):
    io = NpyIO(mpi_tmp_path)
    name = "test_singular"

    io.save_singular(name, array)
    loaded_arr = io.load_singular(name)
    if comm_world.Get_rank() == 0:
        np.testing.assert_equal(loaded_arr, array)
    else:
        assert loaded_arr is None


def test_load_replicated(mpi_tmp_path, array, comm_world):
    io = NpyIO(mpi_tmp_path)
    name = "test_replicated"

    io.save_singular(name, array)
    loaded = io.load_replicated(name)
    np.testing.assert_equal(loaded, array)


def test_load_singular_missing_file(mpi_tmp_path, comm_world):
    io = NpyIO(mpi_tmp_path)
    with pytest.raises(FileNotFoundError):
        io.load_singular("non_existent")


def test_load_replicated_missing_file(mpi_tmp_path, comm_world):
    io = NpyIO(mpi_tmp_path)
    with pytest.raises(FileNotFoundError):
        io.load_replicated("non_existent")
