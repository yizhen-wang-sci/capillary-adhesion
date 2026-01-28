"""
Tests for parameter sweep expansion.
"""

import copy

import pytest
import numpy as np

from a_package.simulation.sweep import unroll_sweep


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
