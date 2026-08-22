"""Tests for the provenance metadata helpers."""

import subprocess
from unittest import mock

import pytest

from a_package.simulation import metadata


def test_git_hash_of_a_clean_package_carries_no_suffix():
    completed = [
        subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        subprocess.CompletedProcess(args=[], returncode=0, stdout="abc123\n", stderr=""),
    ]
    with mock.patch.object(subprocess, "run", side_effect=completed):
        assert metadata.get_git_hash() == "abc123"


def test_git_hash_of_a_modified_package_is_marked_dirty(caplog):
    completed = [
        subprocess.CompletedProcess(args=[], returncode=0, stdout=" M a_package/domain/io.py\n", stderr=""),
        subprocess.CompletedProcess(args=[], returncode=0, stdout="abc123\n", stderr=""),
    ]
    with mock.patch.object(subprocess, "run", side_effect=completed):
        assert metadata.get_git_hash() == "abc123-dirty"
    assert "Uncommitted changes" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(returncode=128, cmd=["git"]),
        subprocess.TimeoutExpired(cmd=["git"], timeout=30),
        FileNotFoundError("git"),
    ],
)
def test_git_hash_is_none_when_a_command_fails(error, caplog):
    with mock.patch.object(subprocess, "run", side_effect=error):
        assert metadata.get_git_hash() is None
    assert "Failed to retrieve Git commit hash" in caplog.text
