"""Numeric arrays kept as `.npy` files."""

import pathlib
from collections.abc import Sequence

import NuMPI.IO
import numpy as np
from NuMPI import MPI


class NpyIO:
    """Save / load a numpy array to / from a npy file, parallel aware."""

    def __init__(
        self,
        domain_shape: Sequence[int] | None = None,
        owned_shape: Sequence[int] | None = None,
        owned_offset: Sequence[int] | None = None,
        communicator: MPI.Intracomm = MPI.COMM_SELF,
    ):
        """Fix how a decomposed array is spread over the ranks, and who takes part.

        Args:
            domain_shape: Shape of the complete array, the one that lands in the file. Left
                unset, along with the other two, for undecomposed data where every rank holds
                the whole array.
            owned_shape: Shape of the part this rank is the authority for. Ghost layers are
                not part of it, so the parts of all ranks tile the domain exactly.
            owned_offset: Where that part begins, in the index space of the domain. The same
                point is the origin of the array this rank hands over.
            communicator: Communicator spanning the ranks taking part. Defaults to
                `MPI.COMM_SELF`, so a parallel run must pass a communicator explicitly.

        Raises:
            ValueError: If the three do not describe one decomposition. Raised on every rank,
                whichever rank found it.
        """
        self._domain_shape = None if domain_shape is None else tuple(domain_shape)
        self._owned_shape = None if owned_shape is None else tuple(owned_shape)
        self._owned_offset = None if owned_offset is None else tuple(owned_offset)
        self._comm = communicator
        self._sync_any_error(self._justify_layout())

    def _justify_layout(self) -> ValueError | None:
        """The error in the three parts of the layout, if there is one."""
        layout = (self._domain_shape, self._owned_shape, self._owned_offset)
        if all(part is None for part in layout):
            return None
        if any(part is None for part in layout):
            return ValueError(
                f"domain_shape, owned_shape and owned_offset describe one decomposition together, "
                f"so they are given together or not at all; got {layout}."
            )

        if (len(self._domain_shape) != len(self._owned_shape)) or (len(self._domain_shape) != len(self._owned_offset)):
            return ValueError(f"The layout must agree in number of dimensions; got {layout}.")
        if any(extent < 1 for extent in self._owned_shape):
            return ValueError(f"Every rank owns at least one point along each dimension; got {self._owned_shape}.")
        if any(start < 0 for start in self._owned_offset):
            return ValueError(f"An owned part begins inside the domain; got offset {self._owned_offset}.")

        over = [
            (start, extent, whole)
            for start, extent, whole in zip(self._owned_offset, self._owned_shape, self._domain_shape)
            if start + extent > whole
        ]
        if len(over):
            return ValueError(
                f"An owned part ends inside the domain, but offset + owned shape passes the domain "
                f"shape along {over} (as offset, owned, domain)."
            )
        return None

    def _sync_error(self, error: Exception | None):
        """Broadcast rank 0's error, if any, and raise it on every rank."""
        error = self._comm.bcast(error, root=0)
        if error is not None:
            raise error

    def _sync_any_error(self, error: Exception | None):
        """Collect the errors of all ranks, and raise the first one on every rank."""
        for gathered in self._comm.allgather(error):
            if gathered is not None:
                raise gathered

    def load_distributed(self, path: pathlib.Path):
        """Read the subdomain of a decomposed array belonging to this rank.

        Args:
            path: File to read.

        Returns:
            This rank's subdomain, shaped as the decomposition prescribes.

        Raises:
            FileNotFoundError: If the file is missing for any rank, raised on every rank
                before the collective read begins.
        """
        self._sync_any_error(None if path.is_file() else FileNotFoundError(f"No file {path}"))
        return NuMPI.IO.load_npy(path, self._owned_offset, self._owned_shape, comm=self._comm)

    def save_distributed(self, path: pathlib.Path, data: np.ndarray):
        """Write a decomposed array, each rank contributing its subdomain.

        Args:
            path: File to write.
            data: This rank's subdomain. Made contiguous before writing.

        Raises:
            FileNotFoundError: If the directory to write into is missing for any rank, raised
                on every rank before the collective write begins.
        """
        self._sync_any_error(None if path.parent.is_dir() else FileNotFoundError(f"No directory {path.parent}"))
        NuMPI.IO.save_npy(
            path,
            np.ascontiguousarray(data),
            self._owned_offset,
            self._domain_shape,
            comm=self._comm,
        )

    def load_singular(self, path: pathlib.Path) -> np.ndarray | None:
        """Read an array on rank 0 only.

        Args:
            path: File to read.

        Returns:
            The array on rank 0, None on every other rank.

        Raises:
            Exception: Whatever `numpy.load` raised on rank 0, re-raised on every rank.
        """
        data, error = None, None
        if self._comm.rank == 0:
            try:
                data = np.load(path, allow_pickle=False)
            except Exception as e:  # noqa: BLE001
                error = e
        self._sync_error(error)
        return data

    def save_singular(self, path: pathlib.Path, data: np.ndarray):
        """Write an array from rank 0 only.

        Args:
            path: File to write.
            data: The array to write. Read on rank 0 alone.

        Raises:
            Exception: Whatever `numpy.save` raised on rank 0, re-raised on every rank.
        """
        error = None
        if self._comm.rank == 0:
            try:
                np.save(path, data)
            except Exception as e:  # noqa: BLE001
                error = e
        self._sync_error(error)

    def load_replicated(self, path: pathlib.Path) -> np.ndarray:
        """Read the same whole array on every rank.

        Args:
            path: File to read.

        Returns:
            The array, identical on every rank.

        Raises:
            Exception: Whatever `numpy.load` raised on any rank, re-raised on every rank.
        """
        data, error = None, None
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as e:  # noqa: BLE001
            error = e
        self._sync_any_error(error)
        return data

    def is_writer(self) -> bool:
        """Whether this is the process that writes when only one process may write."""
        return self._comm.rank == 0

    def is_decomposed(self) -> bool:
        """Whether this rank owns less than the domain."""
        return self._owned_shape != self._domain_shape

    def barrier(self):
        """Wait until every process taking part has arrived."""
        self._comm.Barrier()
