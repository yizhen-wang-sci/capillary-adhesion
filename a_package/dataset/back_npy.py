"""Quantities kept as a directory of `.npy` files."""

import contextlib
import json
import os
import pathlib
from collections.abc import Sequence

import NuMPI.IO
import numpy as np
from NuMPI import MPI

from .quantity import BASIS, Quantity, QuantityBack, QuantityError, Scale


class NpyIO:
    """Filesystem access for a group of processes, coordinated to agree on errors."""

    _ROOT = 0
    """The rank that touches the filesystem where an operation is non-collective.

    Note:
        Don't write any decision within `if self._comm.rank == self._ROOT`, it can lead to 
        processes diverging and eventually hanging.
    """

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
        with self.agreeing_on_error():
            self._check_layout()

    def _check_layout(self):
        """Refuse three parts that are no decomposition.

        Raises:
            ValueError: If the three do not describe one decomposition.
        """
        layout = (self._domain_shape, self._owned_shape, self._owned_offset)
        if all(part is None for part in layout):
            return
        if any(part is None for part in layout):
            raise ValueError(
                f"domain_shape, owned_shape and owned_offset describe one decomposition together, "
                f"so they are given together or not at all; got {layout}."
            )

        if (len(self._domain_shape) != len(self._owned_shape)) or (len(self._domain_shape) != len(self._owned_offset)):
            raise ValueError(f"The layout must agree in number of dimensions; got {layout}.")
        if any(extent < 1 for extent in self._owned_shape):
            raise ValueError(f"Every rank owns at least one point along each dimension; got {self._owned_shape}.")
        if any(start < 0 for start in self._owned_offset):
            raise ValueError(f"An owned part begins inside the domain; got offset {self._owned_offset}.")

        over = [
            (start, extent, whole)
            for start, extent, whole in zip(self._owned_offset, self._owned_shape, self._domain_shape)
            if start + extent > whole
        ]
        if len(over):
            raise ValueError(
                f"An owned part ends inside the domain, but offset + owned shape passes the domain "
                f"shape along {over} (as offset, owned, domain)."
            )

    @contextlib.contextmanager
    def agreeing_on_error(self):
        """Raise on every rank whatever the body raised on any.

        Yields:
            Nothing.

        Raises:
            Exception: Whatever the body raised on the lowest rank that raised.
        """
        error = None
        try:
            yield
        # "except" because all processes must proceed to the next part;
        # while "finally" is only executed when the body exits normally.
        except Exception as e:  # noqa: BLE001
            error = e

        local_state = np.array(self._comm.size, dtype=np.uint32)
        if error is not None:
            local_state[...] = self._comm.rank

        domain_state = np.array(self._comm.size, dtype=np.uint32)
        self._comm.Allreduce(local_state, domain_state, op=MPI.MIN)
        if domain_state < self._comm.size:
            raise self._comm.bcast(error, root=domain_state.item())

    def make_dir(self, path: pathlib.Path):
        """Make a directory, and the parents it needs, on the root rank.

        Args:
            path: Directory to make.

        Raises:
            OSError: Whatever making it raised on the root rank, re-raised on every rank.
        """
        with self.agreeing_on_error():
            if self._comm.rank == self._ROOT:
                path.mkdir(parents=True, exist_ok=True)

    def read_data(self, path: pathlib.Path, decomposed: bool = False) -> np.ndarray:
        """Read an array from one file.

        Args:
            path: File to read.
            decomposed: Whether the array is spread over the ranks.

        Returns:
            This rank's subdomain where the array is decomposed, shaped as the decomposition
            prescribes; the whole array, the same on every rank, where it is not.

        Raises:
            FileNotFoundError: Where the array is decomposed, if the file is missing for any
                rank, raised on every rank before the collective read begins.
            Exception: Where it is not, whatever `numpy.load` raised on the root rank,
                re-raised on every rank.
        """
        if decomposed:
            with self.agreeing_on_error():
                if not path.is_file():
                    raise FileNotFoundError(f"No file {path}")
            return NuMPI.IO.load_npy(path, self._owned_offset, self._owned_shape, comm=self._comm)

        # else
        data = None
        with self.agreeing_on_error():
            if self._comm.rank == self._ROOT:
                # np.load can raise FileNotFoundError
                data = np.load(path, allow_pickle=False)
        return self._comm.bcast(data, root=self._ROOT)

    def write_data(self, path: pathlib.Path, data: np.ndarray, decomposed: bool = False):
        """Write an array to one file.

        Args:
            path: File to write.
            data: This rank's subdomain where the array is decomposed, made contiguous before
                writing; the whole array, taken from the root rank alone, where it is not.
            decomposed: Whether the array is spread over the ranks.

        Raises:
            FileNotFoundError: Where the array is decomposed, if the directory to write into is
                missing for any rank, raised on every rank before the collective write begins.
            Exception: Where it is not, whatever `numpy.save` raised on the root rank,
                re-raised on every rank.
        """
        if decomposed:
            with self.agreeing_on_error():
                if not path.parent.is_dir():
                    raise FileNotFoundError(f"No directory at parent: {path.parent}")
            NuMPI.IO.save_npy(
                path,
                np.ascontiguousarray(data),
                self._owned_offset,
                self._domain_shape,
                comm=self._comm,
            )
            return

        # else
        with self.agreeing_on_error():
            if self._comm.rank == self._ROOT:
                # np.save can raise FileNotFoundError
                np.save(path, data)

    def read_text(self, path: pathlib.Path) -> str:
        """Read a text file, the same content on every rank.

        Args:
            path: File to read.

        Returns:
            What the file holds.

        Raises:
            Exception: Whatever reading it raised on the root rank, re-raised on every rank.
        """
        text = None
        with self.agreeing_on_error():
            if self._comm.rank == self._ROOT:
                text = path.read_text(encoding="utf-8")
        return self._comm.bcast(text, root=self._ROOT)

    def write_text(self, path: pathlib.Path, text: str):
        """Write a text file, in one step, on the root rank alone.

        Args:
            path: File to write.
            text: What the file is to hold.

        Raises:
            Exception: Whatever writing it raised on the root rank, re-raised on every rank.
        """
        with self.agreeing_on_error():
            if self._comm.rank == self._ROOT:
                beside = path.with_name(path.name + "~")
                beside.write_text(text, encoding="utf-8")
                os.replace(beside, path)

    def is_decomposed(self) -> bool:
        """Whether this rank owns less than the domain."""
        return self._owned_shape != self._domain_shape


class NpyBackError(QuantityError):
    """Error due to the limitation of NpyBack implementation, rather than the Quantity."""


class NpyBack(QuantityBack):
    """Values stored as ".npy" files, with a separate startup file to store others.

    Note:
        Because NpyIO only read/write the whole file and in parallel execution, it distributes each
        rank its own share, the bases spread across ranks cannot be indexed. Which bases those are
        must be acknowledged at construction (via `decomposed`) and is then internalised as a
        convention that, those bases, if present in frame of a quantity, must come last.
    """

    def __init__(
        self,
        base_dir: str | os.PathLike,
        io: NpyIO | None = None,
        decomposed: Sequence[str] = (),
    ):
        """Open a directory of `.npy` files, creating it if it is not there yet.

        Args:
            base_dir: Directory where the files are kept.
            io: The parallel-aware file IO, default to no decomposition and `MPI.COMM_SELF`.
            decomposed: Names of the basis quantities spread across ranks, so they cannot be
                indexed. Their names are also written down as the trailing convention of the
                whole directory, which every later session reads back.
                - For simulation (in parallel), it must be the dimensions the grid decomposes.
                - For visualisation (in serial), leave it empty.

        Raises:
            NpyBackError: If the decomposed bases are not in ascending order, if an io that
                hands out shares is given none of them, or if they would expand the trailing
                convention with a quantity already written down.
            OSError: If any rank cannot make `base_dir`.
        """
        self._base_dir = pathlib.Path(base_dir)
        self._io = io if io is not None else NpyIO()
        self._decomposed = sorted(decomposed)
        with self._io.agreeing_on_error():
            if self._decomposed != list(decomposed):
                raise NpyBackError(f"Decomposed bases must be given in ascending order: {decomposed}.")

            # An io that hands out shares must be told which bases it spreads them along
            if self._io.is_decomposed() and not self._decomposed:
                raise NpyBackError("The bases this io decomposes along must be named in `decomposed`; none were.")

        self._io.make_dir(self._base_dir)

        # Load from startup
        startup = self._read_startup()
        self._quantities = startup["quantities"]
        self._trailing_names = startup["frame_convention"]["trailing"]

        # If a new name is specified in decomposed bases, allow expanding the trailing names if
        # it is a new quantity. Always sorted to prevent ambiguous cases.
        new = set(decomposed) - set(self._trailing_names)
        if len(new):
            if new & set(self._quantities.keys()):
                raise NpyBackError(f"Cannot expand trailing bases with existing quantities: {new}.")
            self._trailing_names = sorted(set(self._trailing_names) | new)
            self._write_startup(self._quantities)

    # =========================================================================
    # Startup file

    def _startup_path(self):
        """Path of the file whose content is loaded at instance construction."""
        return self._base_dir / "startup.json"

    @staticmethod
    def _to_written_startup(trailing, quantities):
        """Written down form of the startup."""
        return {"frame_convention": {"trailing": list(trailing)}, "quantities": dict(quantities)}

    def _read_startup(self):
        """Read the startup file, blank where the directory holds none yet."""
        try:
            return json.loads(self._io.read_text(self._startup_path()))
        except FileNotFoundError:
            return self._to_written_startup([], {})

    def _write_startup(self, quantities):
        """Write the startup file.

        Args:
            quantities: The written down form of every quantity, keyed by name.
        """
        startup = self._to_written_startup(self._trailing_names, quantities)
        self._io.write_text(self._startup_path(), json.dumps(startup, indent=2, sort_keys=True))

    # =========================================================================
    # Quantity

    def _check_frame_convention(self, quantity: Quantity):
        """Check whether the frame indeed put the trailing bases at last."""
        # Find the first basis in trailing names
        try:
            i_first = next(
                i_basis for i_basis, basis in enumerate(quantity.frame) if basis.name in self._trailing_names
            )
        except StopIteration:
            return

        # Verify all following ones are also in the trailing names
        if any(basis.name not in self._trailing_names for basis in quantity.frame[i_first + 1 :]):
            raise NpyBackError(
                f"A frame puts the trailing bases {self._trailing_names} last; the frame of {quantity.name} "
                f"is {[basis.name for basis in quantity.frame]}."
            )

    @staticmethod
    def _to_written_unit(unit: str | Scale | None):
        """Written down form of a unit, keeping a literal and a scale apart.

        Args:
            unit: What one unit of the quantity is measured against.

        Returns:
            The literal or the exponent per quantity name, as JSON holds them.

        Raises:
            NpyBackError: If an exponent is neither an integer nor a float.
        """
        if unit is None:
            return None
        if not isinstance(unit, Scale):
            return {"literal": unit}

        exponents = {}
        for name in unit:
            exponent = unit[name]
            if not isinstance(exponent, (int, float)):
                raise NpyBackError(f"An exponent is written down as an integer or a float; got {exponent!r}.")
            exponents[name] = exponent
        return {"scale": exponents}

    def new_quantity(self, new: Quantity):
        """Write down a new quantity."""
        self._check_frame_convention(new)
        record = {
            "unit": self._to_written_unit(new.unit),
            "frame": [basis.name for basis in new.frame],
        }
        # Write down before added in dict to prevent divergence
        self._write_startup(self._quantities | {new.name: record})
        self._quantities[new.name] = record

    def get_all_quantities(self) -> dict[str, Quantity]:
        """Rebuild every quantity written down.

        Returns:
            Keyed by name.

        Raises:
            QuantityError: Cyclic reference or referring to undefined quantity.
        """
        built: dict[str, Quantity] = {}
        building: set[str] = set()

        def build(name: str):
            """Build the named quantity. Recurse when referring to another quantity."""
            if name in built:
                return built[name]

            # Prevent cyclic reference
            if name in building:
                raise QuantityError(f"{name} refers eventually to itself.")
            # Get the unit and frame description
            try:
                record = self._quantities[name]
            except KeyError:
                raise QuantityError(f"{name} is referred but not defined.") from None

            # Build recursively: the quantities it refers to in unit or frame, then itself.
            building.add(name)
            unit = record["unit"]
            if unit is not None:
                if "literal" in unit:
                    unit = unit["literal"]
                else:
                    unit = Scale(unit["scale"])
            frame = tuple(BASIS if basis_name == BASIS.name else build(basis_name) for basis_name in record["frame"])
            built[name] = Quantity(name, unit, frame)
            building.discard(name)

            return built[name]

        for name in self._quantities:
            build(name)
            # Check again because hand-edited file may not comply
            self._check_frame_convention(built[name])

        return built

    # =========================================================================
    # Value

    def _split_address(self, quantity: Quantity, address: tuple):
        """Split an address into the part naming a file and the part indexing the array in it.

        Args:
            quantity: The quantity being addressed.
            address: One index per basis in the frame.

        Returns:
            Two lists of `(basis, index)` pairs, in frame order. The first names a file; the
            second indexes the array in that file, where a basis taken whole is converted to
            `slice(None)`.

        Raises:
            NpyBackError: If the back doesn't support the quantity to be addressed as given.
        """
        pairs = list(zip(quantity.frame, address))

        # Find the first trailing basis
        i_first = next(
            (i_basis for i_basis, basis in enumerate(quantity.frame) if basis.name in self._trailing_names), None
        )
        if i_first is None:
            # If no trailing basis, all bases are treated as indices for the array.
            file_part, index_part = [], pairs
        else:
            # If any trailing basis, only trailing bases are treated as indices, while other bases
            # form the naming scheme of a file.
            file_part, index_part = pairs[:i_first], pairs[i_first:]

            # file naming scheme must locate one single file
            loose = [basis.name for basis, index in file_part if index is None]
            if len(loose):
                raise NpyBackError(f"{loose} of quantity {quantity.name} name a file, and each needs a point.")

        # The indices for the array shall not index any decomposed bases
        if any(basis.name in self._decomposed and index is not None for basis, index in index_part):
            pinned = [basis.name for basis, index in index_part if index is not None]
            raise NpyBackError(f"{pinned} of quantity {quantity.name} are decomposed, and take no point.")

        return file_part, [(basis, slice(None) if index is None else index) for basis, index in index_part]

    def _locate_file(self, quantity: Quantity, file_part: list):
        """Path of the file holding one block of a quantity, named by index in frame order."""
        stem = quantity.name
        if file_part:
            stem += "--" + "_".join(str(index) for _, index in file_part)
        return self._base_dir / f"{stem}.npy"

    def _has_decomposed(self, index_part: list):
        """Whether the array in the file is spanned by decomposed bases."""
        return any(basis.name in self._decomposed for basis, _ in index_part)

    @staticmethod
    def _blank(shape: list[int], dtype: np.dtype):
        """A block holding no value yet.

        Args:
            shape: How many points each basis of the block holds.
            dtype: What the values are to be.

        Returns:
            NaN where the dtype has one, zero where it has not.
        """
        if np.issubdtype(dtype, np.inexact):
            return np.full(shape, np.nan, dtype=dtype)
        return np.zeros(shape, dtype=dtype)

    def _check_ndim(self, quantity: Quantity, address: tuple, value: np.ndarray):
        """Refuse a value whose number of dimensions cannot be covered by the address.

        Args:
            quantity: The quantity being written.
            address: One index per basis in the frame.
            value: What is about to be written.

        Raises:
            QuantityError: If the value has the wrong number of dimensions.
        """
        expected = sum(1 for entry in address if entry is None)
        if value.ndim != expected:
            raise QuantityError(
                f"The frame of {quantity.name} is {[basis.name for basis in quantity.frame]}; its address leaves "
                f"{expected} of them whole, so the value wants {expected} dimensions, got {value.ndim}"
            )

    def save_value(self, quantity: Quantity, address: tuple, value):
        """Write a quantity's value at one place in its frame.

        Args:
            quantity: The quantity, as it was written down.
            address: One index per basis in the frame, `None` where it is taken whole.
            value: This process's part of the block.

        Raises:
            NpyBackError: If the address does not name exactly one file, or indexes a
                decomposed basis.
            QuantityError: If the value has the wrong number of dimensions.
        """
        value = np.asarray(value)
        self._check_ndim(quantity, address, value)
        file_part, index_part = self._split_address(quantity, address)

        path = self._locate_file(quantity, file_part)
        decomposed = self._has_decomposed(index_part)
        subscript = tuple(index for _, index in index_part)

        if all(index == slice(None) for index in subscript):
            self._io.write_data(path, value, decomposed)
            return

        # NpyIO always writes the whole file, so it has to read the whole array and change.
        try:
            whole = self._io.read_data(path, decomposed)
        except FileNotFoundError:
            whole = self._blank([self._length(basis) for basis, _ in index_part], value.dtype)
        whole[subscript] = value
        self._io.write_data(path, whole, decomposed)

    def load_value(self, quantity: Quantity, address: tuple):
        """Read a quantity's value at one place in its frame.

        Args:
            quantity: The quantity, as it was written down.
            address: One index per basis in the frame, `None` where it is taken whole.

        Returns:
            This process's part of the block. A place never written holds NaN where the dtype
            has one, zero where it has not.

        Raises:
            NpyBackError: If the address does not name exactly one file, or indexes a
                decomposed basis.
            QuantityError: If it holds no value there yet.
        """
        file_part, index_part = self._split_address(quantity, address)

        try:
            whole = self._io.read_data(self._locate_file(quantity, file_part), self._has_decomposed(index_part))
        except FileNotFoundError as err:
            raise QuantityError(f"{quantity.name} holds no value at {address} yet") from err

        return whole[tuple(index for _, index in index_part)] if index_part else whole

    def _length(self, basis: Quantity):
        """How many points a basis holds."""
        try:
            return len(self._io.read_data(self._locate_file(basis, [])))
        except FileNotFoundError as err:
            raise QuantityError(f"{basis.name} must have its value saved before it can span anything.") from err
