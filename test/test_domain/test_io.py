"""
Tests of the `storing.py` file.
"""
import os

import pytest
import numpy as np
import numpy.random as random

from a_package.domain import Grid, NpyIO


rng = random.default_rng()


@pytest.fixture
def test_directory():
    path = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(path, exist_ok=True)
    return path


def test_save_load_array(test_directory):
    io = NpyIO(test_directory)
    name = "test_array"
    array = rng.random(10, dtype=float)

    io.save_value_array(name, array)
    loaded_arr = io.load_value_array(name)
    np.testing.assert_equal(loaded_arr, array)


def test_save_load_field(test_directory):
    field_shape = (4, 5)
    grid = Grid([1., 1.], field_shape)
    io = NpyIO(test_directory)
    field = rng.random((2, 3, *field_shape), dtype=float)
    name = "test_field"

    io.save_field(grid, name, field)
    loaded_arr = io.load_field(grid, name)
    np.testing.assert_equal(loaded_arr, field)
