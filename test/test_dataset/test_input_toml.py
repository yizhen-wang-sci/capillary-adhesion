"""Tests for a record's input kept as a TOML file."""

import pytest

from a_package.dataset.input_toml import read_input, write_input


def test_an_input_round_trips_through_the_file(tmp_path):
    config = {
        "solver": {"tolerance": 1e-8, "max_steps": 100, "verbose": True},
        "grid": {"nb_pts": [64, 64], "spacing": 0.5},
        "label": "theta=30",
    }
    path = tmp_path / "input"
    write_input(path, config)
    assert read_input(path) == config


def test_a_second_write_leaves_nothing_of_the_first(tmp_path):
    path = tmp_path / "input"
    write_input(path, {"solver": {"tolerance": 1e-8}})
    write_input(path, {"grid": {"nb_pts": [8, 8]}})
    assert read_input(path) == {"grid": {"nb_pts": [8, 8]}}


def test_reading_an_input_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_input(tmp_path / "absent")
