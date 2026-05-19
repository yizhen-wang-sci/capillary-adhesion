import pytest
import numpy as np
from mpi4py import MPI

from a_package.domain.grid import Grid


@pytest.fixture
def comm_world():
    return MPI.COMM_WORLD


@pytest.fixture
def ref_field():
    return np.arange(100).reshape((10, 10))


def test_real_field_x_decomposition(ref_field, comm_world):
    shape = ref_field.shape
    grid = Grid(shape)

    # Decompose along the non-contiguous axis (for NumPy it is axis 0)
    decomposition = grid.decompose((comm_world.Get_size(), 1), nb_ghost_layers=(1, 1), communicator=comm_world)
    collection = decomposition.collection
    field = collection.real_field("test_field", 1)
    field.s[0, 0, ...] = ref_field[tuple(decomposition.icoords)]

    # NumPy vstack stitches the non-contiguous axis properly
    collected = np.vstack(comm_world.allgather(field.s[0, 0, ...]))
    assert np.allclose(collected, ref_field)


def test_real_field_y_decomposition(ref_field, comm_world):
    shape = ref_field.shape
    grid = Grid(shape)

    # Decompose along the contiguous axis (for NumPy it is axis -1)
    decomposition = grid.decompose((1, comm_world.Get_size()), nb_ghost_layers=(1, 1), communicator=comm_world)
    collection = decomposition.collection
    field = collection.real_field("test_field", 1)
    field.s[0, 0, ...] = ref_field[tuple(decomposition.icoords)]

    # NumPy hstack stitches the contiguous axis properly
    collected = np.hstack(comm_world.allgather(field.s[0, 0, ...]))
    assert np.allclose(collected, ref_field)
