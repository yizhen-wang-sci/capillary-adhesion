"""IO for simulation data exchange, built on top of `domain.NpyIO`."""

import numpy as np
from NuMPI import MPI

from a_package.domain import Field, NpyIO


class SimulationIO:
    """Persistence in the terms a simulation thinks in, over `domain.NpyIO`."""

    def __init__(self, store_dir, decomposition=None, communicator=MPI.COMM_SELF):
        """Open a store directory for a simulation's data.

        Args:
            store_dir: Directory to read from and write to.
            decomposition: The decomposition the fields follow. Leave unset for undecomposed
                data.
            communicator: Communicator spanning the ranks taking part, `MPI.COMM_SELF` by
                default.
        """
        self._io = NpyIO(store_dir, decomposition, communicator)

    def save_constant(self, fields: dict[str, Field] | None = None, single_values: dict[str, float] | None = None):
        """Store quantities that hold for the whole simulation.

        Args:
            fields: Fields to write, by name.
            single_values: Scalars to write, by name.
        """
        if fields is None:
            fields = {}
        if single_values is None:
            single_values = {}

        for name, field in fields.items():
            self._io.save_distributed(name, field)

        for name, value in single_values.items():
            self._io.save_singular(name, np.array([value]))

    def load_constant(self, field_names: list[str] | None = None, single_value_names: list[str] | None = None):
        """Read back quantities that hold for the whole simulation.

        Args:
            field_names: Names of the fields to read.
            single_value_names: Names of the scalars to read.

        Returns:
            Keyed by the names asked for. Fields come back as arrays, scalars as floats.
        """
        if field_names is None:
            field_names = []
        if single_value_names is None:
            single_value_names = []

        result = {}

        # For field, each step has its own file
        for name in field_names:
            result[name] = self._io.load_distributed(name)

        # For single values, all steps shares one file
        for name in single_value_names:
            [result[name]] = self._io.load_replicated(name)

        return result

    def save_step(
        self, index: int, fields: dict[str, Field] | None = None, single_values: dict[str, float] | None = None
    ):
        """Store the quantities of one step.

        Args:
            index: Which step. Steps may be written out of order: a single-value file is
                extended as needed, and the steps skipped over are filled with NaN.
            fields: Fields to write, by name. Each goes to its own per-step file.
            single_values: Scalars to write, by name. Each is placed at `index` in the one
                file that name owns.

        Raises:
            ValueError: If `index` is negative.
        """
        if index < 0:
            raise ValueError("Negative indexing is not supported.")

        if fields is None:
            fields = {}
        if single_values is None:
            single_values = {}

        # For field, each step has its own file
        for name, field in fields.items():
            self._io.save_distributed(_format_filename(name, index), field)

        # For single values, all steps share one file
        for name, value in single_values.items():
            try:
                array = self._io.load_singular(name)
            except FileNotFoundError:
                array = np.empty(0)

            if array is not None:
                if array.size <= index:
                    # We need to extend the array.
                    new_array = np.empty(index + 1)
                    new_array[: array.size] = array
                    new_array[array.size : index] = np.nan
                    array = new_array
                array[index] = value
            self._io.save_singular(name, array)

    def load_step(self, index: int, field_names: list[str] | None = None, single_value_names: list[str] | None = None):
        """Read back the quantities of one step.

        Args:
            index: Which step.
            field_names: Names of the fields to read.
            single_value_names: Names of the scalars to read.

        Returns:
            Keyed by the names asked for, each holding that step's value.
        """
        if field_names is None:
            field_names = []
        if single_value_names is None:
            single_value_names = []

        result = {}

        # For field, each step has its own file
        for name in field_names:
            result[name] = self._io.load_distributed(_format_filename(name, index))

        # For single values, all steps shares one file
        for name in single_value_names:
            result[name] = self._io.load_replicated(name)[index]

        return result

    def save_trajectory(
        self, fields: dict[str, list[Field]] | None = None, single_values: dict[str, np.ndarray] | None = None
    ):
        """Store whole trajectories at once.

        Args:
            fields: One list of steps per name, written to one file per step.
            single_values: One array over all steps per name, written to a single file.
        """
        if fields is None:
            fields = {}
        if single_values is None:
            single_values = {}

        result = {}
        # For field, every step is saved in one file.
        for name, traj in fields.items():
            array = _FieldArray(self._io, name)
            for index in range(len(traj)):
                array[index] = traj[index]
        # For single values, a trajectory is saved as one file
        for name, traj in single_values.items():
            result[name] = self._io.save_singular(name, traj)

    def load_trajectory(self, field_names: list[str] | None = None, single_value_names: list[str] | None = None):
        """Read back whole trajectories.

        Args:
            field_names: Names of the fields to read.
            single_value_names: Names of the scalars to read.

        Returns:
            Keyed by the names asked for. A field comes back as a `_FieldArray`, a single
            value as one array holding every step.
        """
        if field_names is None:
            field_names = []
        if single_value_names is None:
            single_value_names = []

        result = {}
        # For field, every step is saved in one file.
        for name in field_names:
            result[name] = _FieldArray(self._io, name)
        # For single values, a trajectory is saved as one file
        for name in single_value_names:
            result[name] = self._io.load_replicated(name)
        return result


class _FieldArray:
    """Lazy-loading array over the per-step files of one field."""

    def __init__(self, io: NpyIO, name: str):
        """Bind the lazy array to a store and a field name.

        Args:
            io: Where the per-step files live.
            name: Name of the field, which its filenames are derived from.
        """
        self._io = io
        self._name = name

    def __getitem__(self, index: int):
        """Load step `index` from its own file.

        Args:
            index: Which step. Negative indexing is not supported.

        Returns:
            That step, as this rank's subdomain.
        """
        # FIXME: in order for __iter__ to work, this shall capture an error
        # and raise it as IndexError.
        return self._io.load_distributed(_format_filename(self._name, index))

    def __setitem__(self, index: int, value):
        """Store `value` as step `index`, in its own file.

        Args:
            index: Which step. Negative indexing is not supported.
            value: This rank's subdomain of that step.
        """
        self._io.save_distributed(_format_filename(self._name, index), value)

    def __len__(self):
        """Number of steps stored, from the highest index found on disk."""
        i_current = -1
        # FIXME: hardcoded name format
        name_prefix = f"{self._name}--"
        for entry in self._io.root_path.iterdir():
            if entry.name.startswith(name_prefix):
                i_update = int(entry.name[len(name_prefix) :].replace(entry.suffix, ""))
                i_current = max(i_current, i_update)
        return i_current + 1


def _format_filename(name: str, index: int | str):
    """Format a filename with step index.

    Args:
        name: Name of the field.
        index: Which step.
    """
    return f"{name}--{index}"
