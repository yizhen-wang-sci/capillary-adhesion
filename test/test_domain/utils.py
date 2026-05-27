from typing import Sequence

import numpy as np


def generate_global_random_field(nb_pts: Sequence[int], communicator):
    field = np.empty(nb_pts, dtype=float)
    if communicator.rank == 0:
        rng = np.random.default_rng()
        field[...] = rng.random(field.shape)
    communicator.Bcast(field, root=0)
    return field


def generate_global_range_field(nb_pts: Sequence[int], communicator):
    field = np.empty(nb_pts, dtype=int)
    if communicator.rank == 0:
        field[...] = np.arange(np.multiply.reduce(nb_pts)).reshape(nb_pts)
    communicator.Bcast(field, root=0)
    return field
