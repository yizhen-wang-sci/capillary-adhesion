"""
IO for persisting fields and arrays.
"""

import pathlib

import numpy as np
import NuMPI.IO
from NuMPI import MPI


_comm = MPI.COMM_WORLD


class NpyIO:
    """
    NumPy-based parallel-aware data persistence.
    """

    def __init__(self, root_path, decomposition=None):
        self._root_path = pathlib.Path(root_path)
        if decomposition is None:
            # NuMPI.IO will treat it as no decomposition
            self._subdomain_locations = None
            self._nb_subdomain_grid_pts = None
            self._nb_domain_grid_pts = None
        else:
            self._subdomain_locations = tuple(decomposition.subdomain_locations)
            self._nb_subdomain_grid_pts = tuple(decomposition.nb_subdomain_grid_pts)
            self._nb_domain_grid_pts = tuple(decomposition.nb_domain_grid_pts)

    def _to_full_path(self, name: str):
        return self._root_path / f"{name}.npy"

    def load_distributed(self, name: str):
        return NuMPI.IO.load_npy(self._to_full_path(name),
                                 self._subdomain_locations,
                                 self._nb_subdomain_grid_pts)

    def save_distributed(self, name: str, data):
        NuMPI.IO.save_npy(self._to_full_path(name),
                          np.ascontiguousarray(data),
                          self._subdomain_locations,
                          self._nb_domain_grid_pts)

    def load_singular(self, name: str):
        if _comm.rank == 0:
            return np.load(self._to_full_path(name), allow_pickle=False)
        return None

    def save_singular(self, name: str, data):
        if _comm.rank == 0:
            np.save(self._to_full_path(name), data)
        _comm.barrier()

    def load_replicated(self, name: str):
        return _comm.bcast(self.load_singular(name))
