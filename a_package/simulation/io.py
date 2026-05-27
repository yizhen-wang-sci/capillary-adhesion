"""
IO for simulation data exchange.

Provides simulation-aware persistence built on top of domain/io.py.
"""

import numpy as np
from NuMPI import MPI

from a_package.domain import Field, NpyIO


class SimulationIO:

    def __init__(self, store_dir, decomposition=None, communicator=MPI.COMM_SELF):
        self._io = NpyIO(store_dir, decomposition, communicator)

    def save_constant(self, fields: dict[str, Field]=None, single_values: dict[str, float]=None):
        if fields is None:
            fields = {}
        if single_values is None:
            single_values = {}

        for name, field in fields.items():
            self._io.save_distributed(name, field)

        for name, value in single_values.items():
            self._io.save_singular(name, np.array([value]))

    def load_constant(self, field_names: list[str]=None, single_value_names: list[str]=None):
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

    def save_step(self, index: int, fields: dict[str, Field]=None, single_values: dict[str, float]=None):
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
                    new_array[:array.size] = array
                    new_array[array.size:index] = np.nan
                    array = new_array
                array[index] = value
            self._io.save_singular(name, array)

    def load_step(self, index: int, field_names: list[str]=None, single_value_names: list[str]=None):
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

    def save_trajectory(self, fields: dict[str, list[Field]]=None, single_values: dict[str, np.ndarray]=None):
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

    def load_trajectory(self, field_names: list[str]=None, single_value_names: list[str]=None):
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
    """
    Lazy-loading array that reads/writes fields on demand.

    Mimics an array interface but each index access triggers file I/O.
    Useful for large trajectories that don't fit in memory.
    """

    def __init__(self, io: NpyIO, name: str):
        self._io = io
        self._name = name

    def __getitem__(self, index: int):
        return self._io.load_distributed(_format_filename(self._name, index))

    def __setitem__(self, index: int, value):
        self._io.save_distributed(_format_filename(self._name, index), value)

    def __len__(self):
        i_current = -1
        # FIXME: hardcoded name format
        name_prefix = f"{self._name}--"
        for entry in self._io.root_path.iterdir():
            if entry.name.startswith(name_prefix):
                i_update = int(entry.name[len(name_prefix):].replace(".npy", ""))
                i_current = max(i_current, i_update)
        return i_current + 1


def _format_filename(name: str, index: int | str):
    """Format a filename with step index."""
    return f"{name}--{index}"
