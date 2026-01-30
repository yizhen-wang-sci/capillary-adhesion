"""
Parallel-aware data persistence.
"""

import pathlib

import numpy as np

import NuMPI.IO
from NuMPI import MPI

from .grid import Grid


_comm = MPI.COMM_WORLD


class NpyIO:
    """
    NumPy-based parallel-aware data persistence.
    """

    def __init__(self, root_path):
        self._root_path = pathlib.Path(root_path)

    def _to_full_path(self, name: str):
        return self._root_path / f"{name}.npy"

    def load_distributed(self, grid: Grid, name: str):
        return NuMPI.IO.load_npy(
            self._to_full_path(name),
            tuple(grid.subdomain_base),
            tuple(grid.nb_elements))

    def save_distributed(self, grid: Grid, name: str, data):
        NuMPI.IO.save_npy(
            self._to_full_path(name),
            np.ascontiguousarray(data),
            tuple(grid.subdomain_base),
            tuple(grid.nb_elements_global))

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

    def save_aggregated(self, name, data):
        self.save_singular(name, _comm.gather(data))
