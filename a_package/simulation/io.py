"""
IO for simulation data exchange.

Provides simulation-aware persistence built on top of domain/io.py.
"""

import numpy as np

from a_package.domain import Grid, Field, NpyIO


class SimulationIO:

    grid: Grid
    io: NpyIO

    def __init__(self, grid, store_dir) -> None:
        self.grid = grid
        self.io = NpyIO(store_dir)

    def save_constant(self, fields: dict[str, Field]={}, single_values: dict[str, float]={}):
        for [name, field] in fields.items():
            self.io.save_field(self.grid, name, field)

        for [name, value] in single_values.items():
            self.io.save_value_array(name, np.array([value]))

    def load_constant(self, field_names: list[str]=[], single_value_names: list[str]=[]):
        result = {}

        # For field, each step has its own file
        for name in field_names:
            result[name] = self.io.load_field(self.grid, name)

        # For single values, all steps shares one file
        for name in single_value_names:
            [result[name]] = self.io.load_value_array(name)

        return result

    def save_step(self, index: int, fields: dict[str, Field]={}, single_values: dict[str, float]={}):
        # For field, each step has its own file
        for [name, field] in fields.items():
            self.io.save_field(self.grid, format_filename(name, index), field)

        # For single values, all steps share one file
        for [name, value] in single_values.items():
            array = self.io.load_value_array(name)
            try:
                array[index] = value
            except IndexError:
                if index == array.size:
                    array = np.append(array, value)
                else:
                    raise ValueError()
            self.io.save_value_array(name, array)

    def load_step(self, index: int, field_names: list[str]=[], single_value_names: list[str]=[]):
        result = {}

        # For field, each step has its own file
        for name in field_names:
            result[name] = self.io.load_field(self.grid, format_filename(name, index))

        # For single values, all steps shares one file
        for name in single_value_names:
            result[name] = self.io.load_value_array(name)[index]

        return result

    def save_trajectory(self, fields: dict[str, list[Field]]={}, single_values: dict[str, np.ndarray]={}):
        result = {}
        # For field, every step is saved in one file.
        for [name, traj] in fields.items():
            array = FieldArray(self.grid, self.io, name)
            for index in range(len(traj)):
                array[index] = traj[index]
        # For single values, a trajectory is saved as one file
        for [name, traj] in single_values.items():
            result[name] = self.io.save_value_array(name, traj)

    def load_trajectory(self, field_names: list[str]=[], single_value_names: list[str]=[]):
        result = {}
        # For field, every step is saved in one file.
        for name in field_names:
            result[name] = FieldArray(self.grid, self.io, name)
        # For single values, a trajectory is saved as one file
        for name in single_value_names:
            result[name] = self.io.load_value_array(name)
        return result


class FieldArray:
    """
    Lazy-loading array that reads/writes fields on demand.

    Mimics an array interface but each index access triggers file I/O.
    Useful for large trajectories that don't fit in memory.
    """

    grid: Grid
    io: NpyIO
    name: str

    def __init__(self, grid: Grid, io: NpyIO, name: str) -> None:
        self.grid = grid
        self.io = io
        self.name = name

    def __getitem__(self, index: int):
        return self.io.load_field(self.grid, format_filename(self.name, index))

    def __setitem__(self, index: int, value):
        self.io.save_field(self.grid, format_filename(self.name, index), value)


def format_filename(name: str, index: int) -> str:
    """Format a filename with step index."""
    return f"{name}--{index}"
