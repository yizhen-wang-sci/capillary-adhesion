"""
Tests for logging utilities.
"""

import logging
import os
import tempfile

import pytest

from a_package.simulation.logging import reset_logging, switch_log_file


@pytest.fixture
def clean_logging():
    """Reset logging state before and after each test."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.level = original_level


def test_reset_logging_clears_handlers(clean_logging):
    """reset_logging clears existing handlers."""
    root = logging.getLogger()
    root.addHandler(logging.NullHandler())
    root.addHandler(logging.NullHandler())
    assert len(root.handlers) >= 2

    reset_logging()

    # Should have exactly one StreamHandler
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)


def test_switch_log_file_adds_file_handler(clean_logging):
    """switch_log_file adds a FileHandler."""
    reset_logging()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_path = f.name

    try:
        switch_log_file(log_path)

        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == log_path
    finally:
        # Close handler before deleting
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.FileHandler):
                h.close()
        os.unlink(log_path)


def test_switch_log_file_removes_previous_file_handler(clean_logging):
    """switch_log_file removes previous file handler."""
    reset_logging()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f1:
        log_path1 = f1.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f2:
        log_path2 = f2.name

    try:
        switch_log_file(log_path1)
        switch_log_file(log_path2)

        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == log_path2
    finally:
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.FileHandler):
                h.close()
        os.unlink(log_path1)
        os.unlink(log_path2)


def test_switch_log_file_writes_log(clean_logging):
    """switch_log_file writes to the log file."""
    reset_logging()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_path = f.name

    try:
        switch_log_file(log_path)
        logger = logging.getLogger("test_module")
        logger.info("test message")

        # Flush handlers
        for h in logging.getLogger().handlers:
            h.flush()

        with open(log_path, "r") as f:
            content = f.read()
        assert "test message" in content
    finally:
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.FileHandler):
                h.close()
        os.unlink(log_path)
