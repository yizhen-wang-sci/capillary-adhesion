import numpy as np
import pytest
from NuMPI import MPI

from a_package.domain import Grid, factorize_closest


@pytest.fixture
def comm_world():
    return MPI.COMM_WORLD


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
def mock_grid():
    return Grid([4, 4], [0.1, 0.1])


@pytest.fixture
def parallel_in_x(comm_world):
    def _decompose_x(grid):
        return grid.decompose((comm_world.Get_size(), 1), [1, 1], communicator=comm_world)

    def _stitch_x(local_field, grid):
        return np.vstack(comm_world.allgather(local_field))

    return _decompose_x, _stitch_x


@pytest.fixture
def parallel_in_y(comm_world):
    def _decompose_y(grid):
        return grid.decompose((1, comm_world.Get_size()), [1, 1], communicator=comm_world)

    def _stitch_y(local_field, grid):
        return np.hstack(comm_world.allgather(local_field))

    return _decompose_y, _stitch_y


@pytest.fixture
def parallel_in_xy(comm_world):
    def _decompose_xy(grid):
        return grid.decompose(factorize_closest(comm_world.Get_size(), 2), [1, 1], communicator=comm_world)

    def _stitch_xy(local_field, grid):
        values = comm_world.allgather(local_field)
        i_starts = comm_world.allgather(grid.decomposition.subdomain_locations)
        nb_pts = comm_world.allgather(grid.decomposition.nb_subdomain_grid_pts)

        stitched = np.empty(grid.nb_domain_grid_pts)
        for value, (ix, iy), (nx, ny) in zip(values, i_starts, nb_pts):
            stitched[ix : ix + nx, iy : iy + ny] = value
        return stitched

    return _decompose_xy, _stitch_xy


@pytest.fixture(params=["parallel_in_x", "parallel_in_y", "parallel_in_xy"])
def decompose_stitch(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def decomposed_grid(mock_grid, decompose_stitch):
    decompose, _ = decompose_stitch
    decompose(mock_grid)
    return mock_grid
