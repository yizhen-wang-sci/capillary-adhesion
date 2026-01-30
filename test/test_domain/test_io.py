"""
Tests of the `storing.py` file.
"""
import numpy as np
import numpy.random as random
import pytest

from a_package.domain import Grid, NpyIO


rng = random.default_rng()


def test_save_load_singular(tmp_path):
    io = NpyIO(tmp_path)
    name = "test_singular"
    array = rng.random(10, dtype=float)

    io.save_singular(name, array)
    loaded_arr = io.load_singular(name)
    np.testing.assert_equal(loaded_arr, array)


def test_save_load_distributed(tmp_path):
    field_shape = (4, 5)
    grid = Grid([1., 1.], field_shape)
    io = NpyIO(tmp_path)
    field = rng.random((2, 3, *field_shape), dtype=float)
    name = "test_distributed"

    io.save_distributed(grid, name, field)
    loaded_arr = io.load_distributed(grid, name)
    np.testing.assert_equal(loaded_arr, field)


def test_load_singular_missing_file(tmp_path):
    io = NpyIO(tmp_path)
    with pytest.raises(FileNotFoundError):
        io.load_singular("nonexistent_file")


def test_load_replicated(tmp_path):
    io = NpyIO(tmp_path)
    name = "test_replicated"
    array = rng.random(5, dtype=float)

    io.save_singular(name, array)
    loaded = io.load_replicated(name)
    np.testing.assert_equal(loaded, array)


def test_save_aggregated(tmp_path):
    io = NpyIO(tmp_path)
    name = "test_aggregated"
    data = rng.random(3, dtype=float)

    io.save_aggregated(name, data)
    loaded = io.load_singular(name)
    # In single-process, gather returns [data], so saved array is [[...]]
    np.testing.assert_equal(loaded[0], data)
