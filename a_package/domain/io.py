"""
IO for persisting fields and arrays.
"""

import pathlib

import numpy as np

from .grid import Grid
from .field import Field, adapt_shape


class NpyIO:
    """
    NumPy-based persistence for fields and arrays.
    """

    root_path: pathlib.Path

    def __init__(self, root_path):
        self.root_path = pathlib.Path(root_path)

    def _to_full_path(self, name: str):
        return self.root_path / f"{name}.npy"

    def load_field(self, grid: Grid, name: str):
        try:
            field = np.load(self._to_full_path(name), allow_pickle=False)
        except FileNotFoundError:
            field = np.atleast_2d([])
        return adapt_shape(field)

    def save_field(self, grid: Grid, name: str, field: Field):
        np.save(self._to_full_path(name), field)

    def load_value_array(self, name: str):
        try:
            array = np.load(self._to_full_path(name), allow_pickle=False)
        except FileNotFoundError:
            array = np.array([])
        return array

    def save_value_array(self, name: str, array: np.ndarray):
        np.save(self._to_full_path(name), array)
