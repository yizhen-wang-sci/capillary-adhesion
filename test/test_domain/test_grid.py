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


def test_real_field_decomposition(ref_field, comm_world):
    shape = (10, 10)
    grid = Grid(shape)

    # Decompose along the non-contiguous axis (for NumPy it is axis 0)
    decomposition = grid.decompose((comm_world.Get_size(), 1), nb_ghost_layers=(1, 1), communicator=comm_world)
    collection = decomposition.collection
    field = collection.real_field("test_field", 1)
    field.s[0, 0, ...] = ref_field[tuple(decomposition.icoords)]

    # NumPy reshape stitches the non-contiguous axis properly
    collected = np.reshape(comm_world.allgather(field.s), shape)
    assert np.allclose(collected, ref_field)
