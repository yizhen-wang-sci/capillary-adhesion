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


def test_save_load_distributed(tmp_path, field, comm_world):
    grid = Grid(field.shape)
    decomposition = grid.decompose(factorize_closest(comm_world.Get_size(), 2), nb_ghost_layers=(1, 1), communicator=comm_world)
    print("A")
    io = NpyIO(tmp_path, decomposition)
    print("B")
    name = "test_distributed"

    io.save_distributed(name, field[*decomposition.icoords])
    print("C")
    loaded_arr = io.load_distributed(name)
    print("D")
    assert_all_array_equal(loaded_arr, field[*decomposition.icoords])


def test_save_load_singular(tmp_path, array, comm_world):
    io = NpyIO(tmp_path)
    name = "test_singular"

    io.save_singular(name, array)
    loaded_arr = io.load_singular(name)
    if comm_world.Get_rank() == 0:
        np.testing.assert_equal(loaded_arr, array)
    else:
        assert loaded_arr is None


def test_load_replicated(tmp_path, array):
    io = NpyIO(tmp_path)
    name = "test_replicated"

    io.save_singular(name, array)
    loaded = io.load_replicated(name)
    np.testing.assert_equal(loaded, array)
