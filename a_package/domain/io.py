"""IO for persisting fields and arrays."""

import pathlib

import numpy as np
import NuMPI.IO
from NuMPI import MPI


class NpyIO:
    """NumPy-based parallel-aware data persistence."""

    def __init__(self, root_path, decomposition=None, communicator=MPI.COMM_SELF):
        """Open a directory of `.npy` files, optionally following a decomposition.

        Args:
            root_path: Directory to read from and write to. Not created here.
            decomposition: The decomposition the distributed arrays follow. Leave unset for
                undecomposed data.
            communicator: Communicator spanning the ranks taking part, `MPI.COMM_SELF` by
                default.
        """
        self.root_path = pathlib.Path(root_path)

        if decomposition is None:
            # NuMPI.IO will treat it as no decomposition
            self._subdomain_locations = None
            self._nb_subdomain_grid_pts = None
            self._nb_domain_grid_pts = None
        else:
            self._subdomain_locations = tuple(decomposition.subdomain_locations)
            self._nb_subdomain_grid_pts = tuple(decomposition.nb_subdomain_grid_pts)
            self._nb_domain_grid_pts = tuple(decomposition.nb_domain_grid_pts)

        self._comm = communicator

    def _to_full_path(self, name: str):
        """Path of the `.npy` file backing `name`."""
        return self.root_path / f"{name}.npy"

    def load_distributed(self, name: str):
        """Read the subdomain of a decomposed array belonging to this rank.

        Args:
            name: Name of the array, without the `.npy` suffix.

        Returns:
            This rank's subdomain, shaped as the decomposition prescribes.

        Raises:
            FileNotFoundError: If the file is missing for any rank, raised on every rank
                before the collective read begins.
        """
        path = self._to_full_path(name)
        self._sync_error_any_rank(None if path.is_file() else FileNotFoundError(f"No file {path}"))
        return NuMPI.IO.load_npy(path, self._subdomain_locations, self._nb_subdomain_grid_pts, comm=self._comm)

    def save_distributed(self, name: str, data):
        """Write a decomposed array, each rank contributing its subdomain.

        Args:
            name: Name of the array, without the `.npy` suffix.
            data: This rank's subdomain. Made contiguous before writing.
        """
        NuMPI.IO.save_npy(
            self._to_full_path(name),
            np.ascontiguousarray(data),
            self._subdomain_locations,
            self._nb_domain_grid_pts,
            comm=self._comm,
        )

    def _sync_error(self, error):
        """Broadcast rank 0's error, if any, and raise it on every rank.

        Args:
            error: The error caught on this rank, or None.

        Raises:
            Exception: Whatever error rank 0 passed in.
        """
        error = self._comm.bcast(error, root=0)
        if error is not None:
            raise error

    def _sync_error_any_rank(self, error):
        """Gather the error of every rank and raise the first one, on every rank.

        Args:
            error: The error caught on this rank, or None.

        Raises:
            Exception: The error of the lowest-numbered rank that caught one.
        """
        for error in self._comm.allgather(error):
            if error is not None:
                raise error

    def load_singular(self, name: str):
        """Read an array on rank 0 only.

        Args:
            name: Name of the array, without the `.npy` suffix.

        Returns:
            The array on rank 0, None on every other rank.

        Raises:
            Exception: Whatever `numpy.load` raised on rank 0, re-raised on every rank.
        """
        data, error = None, None
        if self._comm.rank == 0:
            try:
                data = np.load(self._to_full_path(name), allow_pickle=False)
            except Exception as e:
                error = e
        self._sync_error(error)
        return data

    def save_singular(self, name: str, data):
        """Write an array from rank 0 only.

        Args:
            name: Name of the array, without the `.npy` suffix.
            data: The array to write. Read on rank 0 alone.

        Raises:
            Exception: Whatever `numpy.save` raised on rank 0, re-raised on every rank.
        """
        error = None
        if self._comm.rank == 0:
            try:
                np.save(self._to_full_path(name), data)
            except Exception as e:
                error = e
        self._sync_error(error)

    def load_replicated(self, name: str):
        """Read the same whole array on every rank.

        Args:
            name: Name of the array, without the `.npy` suffix.

        Returns:
            The array, identical on every rank.

        Raises:
            Exception: Whatever `numpy.load` raised on the lowest-numbered rank that
                failed, re-raised on every rank.
        """
        data, error = None, None
        try:
            data = np.load(self._to_full_path(name), allow_pickle=False)
        except Exception as e:
            error = e
        self._sync_error_any_rank(error)
        return data
