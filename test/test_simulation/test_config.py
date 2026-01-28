"""
Tests for TOML configuration loading.
"""

import os
import tempfile

import pytest

from a_package.simulation.config import load_config, save_config, backfill_config


@pytest.fixture
def sample_toml_content():
    return """
[section_a]
key1 = "value1"
key2 = 42

[section_a.nested]
flag = true
ratio = 3.14

[section_b]
items = [1, 2, 3]
"""


@pytest.fixture
def temp_toml_file(sample_toml_content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(sample_toml_content)
        filepath = f.name
    yield filepath
    os.unlink(filepath)


def test_load_config(temp_toml_file):
    """Load TOML and verify structure."""
    config = load_config(temp_toml_file)

    assert isinstance(config, dict)
    assert config["section_a"]["key1"] == "value1"
    assert config["section_a"]["key2"] == 42
    assert config["section_a"]["nested"]["flag"] is True
    assert config["section_a"]["nested"]["ratio"] == 3.14
    assert config["section_b"]["items"] == [1, 2, 3]


def test_save_and_reload_config(temp_toml_file):
    """Save and reload preserves content."""
    config = load_config(temp_toml_file)

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
        output_path = f.name

    try:
        save_config(config, output_path)
        reloaded = load_config(output_path)
        assert reloaded == config
    finally:
        os.unlink(output_path)


def test_load_config_merge():
    """Later files override earlier."""
    base = """
[section]
a = 1
b = 2
"""
    override = """
[section]
b = 99
c = 3
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f1:
        f1.write(base)
        base_path = f1.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f2:
        f2.write(override)
        override_path = f2.name

    try:
        config = load_config(base_path, override_path)
        assert config["section"]["a"] == 1   # from base
        assert config["section"]["b"] == 99  # overridden
        assert config["section"]["c"] == 3   # from override
    finally:
        os.unlink(base_path)
        os.unlink(override_path)


def test_backfill_config():
    """Backfill fills missing fields from defaults."""
    config = {"a": {"b": 1}}
    defaults = {"a": {"b": 0, "c": 2}, "d": 3}

    result = backfill_config(config, defaults)

    assert result["a"]["b"] == 1  # config takes precedence
    assert result["a"]["c"] == 2  # filled from defaults
    assert result["d"] == 3       # filled from defaults
