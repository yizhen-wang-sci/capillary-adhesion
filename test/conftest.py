import logging

import pytest
from NuMPI import MPI

from a_package.simulation import setup_logging


@pytest.fixture(autouse=True)
def _configure_logging(tmp_path):
    setup_logging(level=logging.DEBUG)


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
