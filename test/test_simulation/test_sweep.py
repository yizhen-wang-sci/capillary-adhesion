"""
Tests for parameter sweep expansion.
"""

import copy

import pytest
import numpy as np

from a_package.simulation.sweep import size_of_sweep, unroll_sweep


@pytest.fixture
def config_no_sweep():
    return {
        "problem": {"capillary": {"contact_angle": 45.0}},
        "solver": {"tolerance": 1e-6},
    }


@pytest.fixture
def config_single_sweep():
    return {
        "problem": {"capillary": {"contact_angle": 45.0}},
        "solver": {"tolerance": 1e-6},
        "sweep": [
            {"path": "problem.capillary.contact_angle", "linspace": [30.0, 90.0, 4]},
        ],
    }


@pytest.fixture
def config_multiple_sweeps():
    return {
        "problem": {"capillary": {"contact_angle": 45.0}},
        "solver": {"tolerance": 1e-6},
        "sweep": [
            {"path": "problem.capillary.contact_angle", "values": [30.0, 60.0]},
            {"path": "solver.tolerance", "logspace": [-6, -4, 3]},
        ],
    }


def test_unroll_sweep_no_sweep(config_no_sweep):
    """No sweep defined - yields config once unchanged."""
    original = copy.deepcopy(config_no_sweep)

    results = list(unroll_sweep(config_no_sweep))

    assert len(results) == 1
    assert "sweep" not in results[0]
    assert results[0]["problem"]["capillary"]["contact_angle"] == original["problem"]["capillary"]["contact_angle"]


def test_unroll_sweep_single_sweep(config_single_sweep):
    """Single sweep with linspace - yields correct values."""
    results = []
    for config in unroll_sweep(config_single_sweep):
        results.append(config["problem"]["capillary"]["contact_angle"])

    assert len(results) == 4
    np.testing.assert_array_almost_equal(results, [30.0, 50.0, 70.0, 90.0])


def test_unroll_sweep_multiple_sweeps(config_multiple_sweeps):
    """Multiple sweeps - yields Cartesian product."""
    results = []
    for config in unroll_sweep(config_multiple_sweeps):
        results.append((
            config["problem"]["capillary"]["contact_angle"],
            config["solver"]["tolerance"],
        ))

    # 2 angles * 3 tolerances = 6 combinations
    assert len(results) == 6

    # Check all combinations present
    angles = {r[0] for r in results}
    tolerances = {r[1] for r in results}
    assert angles == {30.0, 60.0}
    np.testing.assert_array_almost_equal(sorted(tolerances), [1e-6, 1e-5, 1e-4])


def test_unroll_sweep_mutates_input(config_single_sweep):
    """Verifies that sweep key is popped from input config."""
    assert "sweep" in config_single_sweep

    list(unroll_sweep(config_single_sweep))

    assert "sweep" not in config_single_sweep


def test_size_of_sweep_without_a_sweep_is_one(config_no_sweep):
    """A config with no sweep yields one combination."""
    assert size_of_sweep(config_no_sweep) == 1


def test_size_of_sweep_counts_the_cartesian_product(config_multiple_sweeps):
    """Two values times three values."""
    assert size_of_sweep(config_multiple_sweeps) == 6


def test_size_of_sweep_leaves_the_config_alone(config_single_sweep):
    """The sweep key stays in place."""
    size_of_sweep(config_single_sweep)
    assert "sweep" in config_single_sweep


def test_duplicated_paths_are_refused():
    """Two sweeps over one path."""
    config = {
        "solver": {"tolerance": 1e-6},
        "sweep": [
            {"path": "solver.tolerance", "values": [1e-6]},
            {"path": "solver.tolerance", "values": [1e-5]},
        ],
    }
    with pytest.raises(ValueError, match="Duplicated sweeps"):
        size_of_sweep(config)


def test_a_sweep_without_values_is_refused():
    """Neither values, linspace, nor logspace."""
    config = {"solver": {"tolerance": 1e-6}, "sweep": [{"path": "solver.tolerance", "step": 2}]}
    with pytest.raises(ValueError, match="no supported value specification"):
        size_of_sweep(config)


def test_a_path_missing_from_the_config_is_refused():
    """The swept path leads through a key that does not exist."""
    config = {"solver": {}, "sweep": [{"path": "solver.missing.tolerance", "values": [1e-6]}]}
    with pytest.raises(KeyError):
        list(unroll_sweep(config))


def test_linspace_expands_to_evenly_spaced_values():
    config = {"solver": {"tolerance": 0.0}, "sweep": [{"path": "solver.tolerance", "linspace": [0.0, 1.0, 3]}]}
    swept = [config["solver"]["tolerance"] for config in unroll_sweep(config)]
    assert swept == pytest.approx([0.0, 0.5, 1.0])


def test_logspace_expands_to_powers_of_the_endpoints():
    config = {"solver": {"tolerance": 0.0}, "sweep": [{"path": "solver.tolerance", "logspace": [-6, -3, 4]}]}
    swept = [config["solver"]["tolerance"] for config in unroll_sweep(config)]
    assert swept == pytest.approx([1e-6, 1e-5, 1e-4, 1e-3])


@pytest.mark.parametrize(
    ("spec", "expected"),
    [({"values": [1, 2]}, 2), ({"linspace": [0.0, 1.0, 3]}, 3), ({"logspace": [-6, -3, 4]}, 4)],
)
def test_size_of_sweep_counts_each_kind_of_specification(spec, expected):
    config = {"solver": {"tolerance": 0.0}, "sweep": [{"path": "solver.tolerance", **spec}]}
    assert size_of_sweep(config) == expected


def test_a_swept_value_is_a_float_whatever_the_specification():
    config = {"solver": {"tolerance": 0.0}, "sweep": [{"path": "solver.tolerance", "linspace": [0, 1, 2]}]}
    swept = [config["solver"]["tolerance"] for config in unroll_sweep(config)]
    assert all(isinstance(value, float) for value in swept)
