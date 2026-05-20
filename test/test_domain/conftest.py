import pytest

import numpy as np
from NuMPI import MPI


@pytest.fixture
def comm_world():
    return MPI.COMM_WORLD


@pytest.fixture
def ref_field():
    return np.arange(100).reshape((10, 10))
