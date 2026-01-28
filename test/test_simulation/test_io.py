"""
Tests for SimulationIO.
"""

import shutil
import tempfile

import pytest
import numpy as np

from a_package.domain import Grid
from a_package.simulation.io import SimulationIO


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def grid():
    """Create a simple grid."""
    return Grid([1.0, 1.0], [8, 8])


@pytest.fixture
def io(grid, temp_dir):
    """Create a SimulationIO instance."""
    return SimulationIO(grid, temp_dir)


# =============================================================================
# Constants
# =============================================================================


def test_save_load_constant_field(io):
    """Save and load constant field."""
    field = np.random.random((1, 1, 8, 8))

    io.save_constant(fields={"my_field": field})
    result = io.load_constant(field_names=["my_field"])

    np.testing.assert_array_almost_equal(result["my_field"], field)


def test_save_load_constant_single_value(io):
    """Save and load constant single value."""
    io.save_constant(single_values={"my_value": 1.5})
    result = io.load_constant(single_value_names=["my_value"])

    assert result["my_value"] == pytest.approx(1.5)


def test_save_load_constant_mixed(io):
    """Save and load both fields and single values."""
    field = np.random.random((1, 1, 8, 8))

    io.save_constant(fields={"field_a": field}, single_values={"value_a": 2.0})
    result = io.load_constant(field_names=["field_a"], single_value_names=["value_a"])

    np.testing.assert_array_almost_equal(result["field_a"], field)
    assert result["value_a"] == pytest.approx(2.0)


# =============================================================================
# Steps
# =============================================================================


def test_save_load_step_field(io):
    """Save and load step field."""
    field = np.random.random((1, 1, 8, 8))

    io.save_step(0, fields={"field": field})
    result = io.load_step(0, field_names=["field"])

    np.testing.assert_array_almost_equal(result["field"], field)


def test_save_load_step_single_value(io):
    """Save and load step single value."""
    io.save_step(0, single_values={"x": 0.1})
    io.save_step(1, single_values={"x": 0.2})

    result0 = io.load_step(0, single_value_names=["x"])
    result1 = io.load_step(1, single_value_names=["x"])

    assert result0["x"] == pytest.approx(0.1)
    assert result1["x"] == pytest.approx(0.2)


def test_save_load_multiple_steps(io):
    """Save and load multiple steps."""
    fields = [np.random.random((1, 1, 8, 8)) for _ in range(3)]
    values = [0.1, 0.2, 0.3]

    for i, (field, val) in enumerate(zip(fields, values)):
        io.save_step(i, fields={"field": field}, single_values={"val": val})

    for i, (expected_field, expected_val) in enumerate(zip(fields, values)):
        result = io.load_step(i, field_names=["field"], single_value_names=["val"])
        np.testing.assert_array_almost_equal(result["field"], expected_field)
        assert result["val"] == pytest.approx(expected_val)


# =============================================================================
# Trajectory
# =============================================================================


def test_save_load_trajectory_single_values(io):
    """Save and load trajectory single values."""
    arr_a = np.array([0.1, 0.2, 0.3])
    arr_b = np.array([1.0, 2.0, 3.0])

    io.save_trajectory(single_values={"a": arr_a, "b": arr_b})
    result = io.load_trajectory(single_value_names=["a", "b"])

    np.testing.assert_array_almost_equal(result["a"], arr_a)
    np.testing.assert_array_almost_equal(result["b"], arr_b)


def test_load_trajectory_fields_lazy(io):
    """load_trajectory returns lazy-loading FieldArray for fields."""
    fields = [np.random.random((1, 1, 8, 8)) for _ in range(3)]

    for i, field in enumerate(fields):
        io.save_step(i, fields={"field": field})

    result = io.load_trajectory(field_names=["field"])
    field_array = result["field"]

    for i, expected in enumerate(fields):
        np.testing.assert_array_almost_equal(field_array[i], expected)


def test_save_trajectory_fields(io):
    """save_trajectory saves fields by index."""
    fields = [np.random.random((1, 1, 8, 8)) for _ in range(3)]

    io.save_trajectory(fields={"field": fields})
    result = io.load_trajectory(field_names=["field"])

    for i, expected in enumerate(fields):
        np.testing.assert_array_almost_equal(result["field"][i], expected)


# =============================================================================
# Grid access
# =============================================================================


def test_io_grid_property(io, grid):
    """SimulationIO.grid returns the grid."""
    assert io.grid is grid
