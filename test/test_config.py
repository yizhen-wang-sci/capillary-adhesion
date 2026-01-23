"""
Tests for TOML configuration loading and sweep expansion.
"""

import os
import tempfile

import pytest
import numpy as np

from a_package.simulation import (
    load_config,
    save_config,
    Config,
    expand_sweep_spec,
    count_sweep_combinations,
)
from a_package.model.surfaces import generate_surface
from a_package.domain import Grid


@pytest.fixture
def sample_toml_content():
    return """
[domain]
[domain.grid]
pixel_size = 0.05
nb_pixels = 64

[problem]
[problem.upper]
shape = "tip"
radius = 10.0

[problem.lower]
shape = "flat"
constant = 0.0

[problem.capillary]
interface_thickness = 0.05
contact_angle_degree = 45.0

[solver]
[solver.optimizer]
max_nb_iters = 1000
max_nb_loops = 30
tol_convergence = 1e-6
tol_constraints = 1e-8
init_penalty_weight = 0.1

[simulation]
[simulation.trajectory]
type = "approach_retract"
min_separation = 0.0
max_separation = 0.1
step_size = 0.01
round_trip = true

[simulation.constraint]
type = "constant_volume"
liquid_volume_percent = 15.0
"""


@pytest.fixture
def sample_toml_with_sweep():
    return """
[domain]
[domain.grid]
pixel_size = 0.05
nb_pixels = 64

[problem]
[problem.upper]
shape = "tip"
radius = 10.0

[problem.lower]
shape = "flat"
constant = 0.0

[problem.capillary]
interface_thickness = 0.05
contact_angle_degree = 45.0

[solver]
[solver.optimizer]
max_nb_iters = 1000
max_nb_loops = 30
tol_convergence = 1e-6
tol_constraints = 1e-8
init_penalty_weight = 0.1

[simulation]
[simulation.trajectory]
type = "approach_retract"
min_separation = 0.0
max_separation = 0.1
step_size = 0.01
round_trip = true

[simulation.constraint]
type = "constant_volume"
liquid_volume_percent = 15.0

[[sweep]]
path = "simulation.constraint.liquid_volume_percent"
linspace = [20.0, 80.0, 4]
"""


@pytest.fixture
def temp_toml_file(sample_toml_content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(sample_toml_content)
        filepath = f.name
    yield filepath
    os.unlink(filepath)


def test_load_config(temp_toml_file):
    """Test loading a TOML config file."""
    config = load_config(temp_toml_file)

    assert isinstance(config, Config)
    # Domain - only grid
    assert config.domain["grid"]["pixel_size"] == 0.05
    assert config.domain["grid"]["nb_pixels"] == 64
    # Problem - surfaces and capillary
    assert config.problem["upper"]["shape"] == "tip"
    assert config.problem["upper"]["radius"] == 10.0
    assert config.problem["lower"]["shape"] == "flat"
    # Problem capillary is a raw dict
    assert config.problem["capillary"]["contact_angle_degree"] == 45.0
    assert config.problem["capillary"]["interface_thickness"] == 0.05
    # Solver optimizer
    assert config.solver["optimizer"]["max_nb_iters"] == 1000
    # Simulation trajectory
    assert config.simulation["trajectory"]["type"] == "approach_retract"
    assert config.simulation["trajectory"]["step_size"] == 0.01
    assert config.simulation["trajectory"]["round_trip"] == True
    # Simulation constraint
    assert config.simulation["constraint"]["type"] == "constant_volume"
    assert config.simulation["constraint"]["liquid_volume_percent"] == 15.0


def test_save_and_reload_config(temp_toml_file):
    """Test saving and reloading a config."""
    config = load_config(temp_toml_file)

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
        output_path = f.name

    try:
        save_config(config, output_path)
        reloaded = load_config(output_path)

        assert reloaded.domain["grid"]["pixel_size"] == config.domain["grid"]["pixel_size"]
        assert reloaded.problem["upper"]["shape"] == config.problem["upper"]["shape"]
        assert reloaded.simulation["constraint"]["liquid_volume_percent"] == config.simulation["constraint"]["liquid_volume_percent"]
    finally:
        os.unlink(output_path)


def test_expand_sweep_spec_no_sweep(temp_toml_file):
    """Test expansion when no sweeps are defined."""
    config = load_config(temp_toml_file)

    expanded = list(expand_sweep_spec(config.sweep))
    assert len(expanded) == 1
    assert expanded[0] == {}  # Empty overrides dict


def test_expand_sweep_spec_with_linspace(sample_toml_with_sweep):
    """Test sweep expansion with linspace."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(sample_toml_with_sweep)
        filepath = f.name

    try:
        config = load_config(filepath)

        assert len(config.sweep) == 1
        assert count_sweep_combinations(config.sweep) == 4

        expanded = list(expand_sweep_spec(config.sweep))
        assert len(expanded) == 4

        # Check the swept values (now as override dicts)
        path = "simulation.constraint.liquid_volume_percent"
        volumes = [overrides[path] for overrides in expanded]
        np.testing.assert_array_almost_equal(volumes, [20.0, 40.0, 60.0, 80.0])
    finally:
        os.unlink(filepath)


def test_expand_sweep_spec_with_multiple_sweeps():
    """Test sweep expansion with multiple sweep parameters."""
    sweep_spec = [
        {"path": "simulation.constraint.liquid_volume_percent", "linspace": [20.0, 40.0, 3]},
        {"path": "problem.capillary.contact_angle_degree", "values": [30.0, 60.0]},
    ]

    assert count_sweep_combinations(sweep_spec) == 6  # 3 * 2

    expanded = list(expand_sweep_spec(sweep_spec))
    assert len(expanded) == 6

    # Each override dict should have both paths
    for overrides in expanded:
        assert "simulation.constraint.liquid_volume_percent" in overrides
        assert "problem.capillary.contact_angle_degree" in overrides


def test_generate_surface_flat():
    """Test flat surface generation."""
    grid = Grid([1.0, 1.0], [32, 32])
    height = generate_surface(grid, "flat", constant=0.5)

    assert height.shape == (32, 32)
    np.testing.assert_array_almost_equal(height, 0.5 * np.ones((32, 32)))


def test_generate_surface_tip():
    """Test tip surface generation."""
    grid = Grid([1.0, 1.0], [32, 32])
    height = generate_surface(grid, "tip", radius=10.0)

    assert height.shape == (32, 32)
    # Minimum should be at center, value should be 0
    center_idx = 16
    assert height[center_idx, center_idx] == np.min(height)


def test_generate_surface_sinusoid():
    """Test sinusoidal surface generation."""
    grid = Grid([1.0, 1.0], [32, 32])
    height = generate_surface(grid, "sinusoid", wavenumber=2.0, amplitude=0.1)

    assert height.shape == (32, 32)
    assert np.max(height) <= 0.1
    assert np.min(height) >= -0.1


def test_generate_surface_unknown_type():
    """Test that unknown surface type raises error."""
    grid = Grid([1.0, 1.0], [32, 32])

    with pytest.raises(ValueError, match="Unknown surface shape"):
        generate_surface(grid, "unknown_shape")
